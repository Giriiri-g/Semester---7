import os
import math
import time
import shutil
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import chess
import chess.engine


PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        # Input: 18 channels (Tactical + Material planes)
        self.conv_input = nn.Conv2d(18, 128, 3, padding=1)
        self.bn_input = nn.BatchNorm2d(128)
        self.res_blocks = nn.ModuleList([ResidualBlock(128) for _ in range(6)])
        
        # Policy Head
        self.policy_conv = nn.Conv2d(128, 32, 1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, 4096)
        
        # Value Head
        self.value_conv = nn.Conv2d(128, 16, 1)
        self.value_bn = nn.BatchNorm2d(16)
        self.value_fc1 = nn.Linear(16 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.res_blocks:
            x = block(x)
        
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 32 * 8 * 8)
        policy = self.policy_fc(policy)
        
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 16 * 8 * 8)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value



def board_to_tensor(board):
    """Encodes board into 18x8x8 tensor"""
    tensor = np.zeros((18, 8, 8), dtype=np.float32)
    
    piece_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    
    # 1. Piece Positions (0-11)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row, col = divmod(square, 8)
            channel = piece_map[piece.piece_type]
            if not piece.color: # Black
                channel += 6
            tensor[channel, row, col] = 1.0
            
    # 2. Global State (12-13)
    # Castling
    castling_val = 0.0
    if board.has_kingside_castling_rights(board.turn): castling_val += 1.0
    if board.has_queenside_castling_rights(board.turn): castling_val += 0.5
    tensor[12, :, :] = castling_val
    
    # En-passant
    if board.ep_square is not None:
        r, c = divmod(board.ep_square, 8)
        tensor[13, r, c] = 1.0

    # 3. Tactical Planes (14-17)
    us = board.turn
    them = not board.turn
    
    for square in chess.SQUARES:
        row, col = divmod(square, 8)
        
        attacked_by_us = board.is_attacked_by(us, square)
        attacked_by_them = board.is_attacked_by(them, square)
        
        tensor[14, row, col] = 1.0 if attacked_by_us else 0.0
        tensor[15, row, col] = 1.0 if attacked_by_them else 0.0
        
        piece = board.piece_at(square)
        if piece:
            val = PIECE_VALUES.get(piece.piece_type, 0) / 10.0
            # Aggression: higher preference
            if piece.color == them and attacked_by_us:
                tensor[16, row, col] = val*3
            # Defense: normal preference
            if piece.color == us and attacked_by_them:
                tensor[17, row, col] = val

    return torch.FloatTensor(tensor).unsqueeze(0)

def move_to_index(move):
    return int(move.from_square) * 64 + int(move.to_square)

def index_to_move(index, board):
    from_sq = index // 64
    to_sq = index % 64
    move = chess.Move(from_sq, to_sq)
    if move in board.legal_moves: return move
    for p in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
        pm = chess.Move(from_sq, to_sq, promotion=p)
        if pm in board.legal_moves: return pm
    return None


class MCTSNode:
    def __init__(self, board, parent=None, move=None, prior=0.0):
        self.board = board.copy()
        self.parent = parent
        self.move = move
        self.prior = float(prior)
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False

    def value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    def ucb_score(self, c_puct=1.5):
        if self.parent is None: return float('inf')
        parent_visits = max(1, self.parent.visit_count)
        return self.value() + c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)

    def select_child(self):
        return max(self.children.values(), key=lambda n: n.ucb_score())

    def expand(self, policy_probs):
        if self.is_expanded: return
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            self.is_expanded = True
            return
        
        for move in legal_moves:
            idx = move_to_index(move)
            prior = float(policy_probs[idx]) if 0 <= idx < len(policy_probs) else 1e-6
            new_board = self.board.copy()
            new_board.push(move)
            self.children[move] = MCTSNode(new_board, parent=self, move=move, prior=prior)
        self.is_expanded = True

    def backup(self, value):
        self.visit_count += 1
        self.value_sum += value
        if self.parent: self.parent.backup(-value)

class MCTS:
    def __init__(self, agent, num_simulations=200):
        self.agent = agent
        self.num_simulations = num_simulations

    def search(self, board):
        root = MCTSNode(board)
        
        for _ in range(self.num_simulations):
            node = root
            # Selection
            while node.is_expanded and node.children:
                node = node.select_child()
            
            # Expansion & Evaluation
            if not node.board.is_game_over():
                self.agent.model.eval()
                state = board_to_tensor(node.board).to(self.agent.device)
                with torch.no_grad():
                    logits, val = self.agent.model(state)
                    probs = F.softmax(logits[0], dim=0).cpu().numpy()
                    value = float(val.item())
                node.expand(probs)
            else:
                # Terminal state handling
                if node.board.is_checkmate():
                    value = -1.0 
                else:
                    value = 0.0
            
            # Backpropagation
            node.backup(value)
            
        move_probs = {}
        total_visits = sum(c.visit_count for c in root.children.values()) or 1
        for m, c in root.children.items():
            move_probs[m] = c.visit_count / total_visits
        return move_probs, root.value()


class ChessAgent:
    def __init__(self, model_path=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChessNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.mcts_simulations = 100
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            print(f"Loaded model: {model_path}")

    def select_move_mcts(self, board, temperature=1.0):
        mcts = MCTS(self, self.mcts_simulations)
        move_probs, value = mcts.search(board)
        if not move_probs: return None, {}, 0.0
        
        moves = list(move_probs.keys())
        probs = np.array([move_probs[m] for m in moves])
        
        if temperature < 0.1:
            move = moves[np.argmax(probs)]
        else:
            probs = probs ** (1.0 / temperature)
            probs /= probs.sum()
            move = np.random.choice(moves, p=probs)
        return move, move_probs, value

    def select_move_fast(self, board, temperature=1.0):
        self.model.eval()
        with torch.no_grad():
            state = board_to_tensor(board).to(self.device)
            logits, val = self.model(state)
        
        legal_moves = list(board.legal_moves)
        if not legal_moves: return None, {}, 0.0
        
        indices = [move_to_index(m) for m in legal_moves]
        move_logits = logits[0, indices] / max(1e-8, temperature)
        probs = F.softmax(move_logits, dim=0).cpu()
        
        idx = torch.multinomial(probs, 1).item()
        move_dict = {m: float(probs[i]) for i, m in enumerate(legal_moves)}
        return legal_moves[idx], move_dict, float(val.item())

    def train_step(self, states, move_probs_list, values, outcomes):
        self.model.train()
        if not states: return 0, 0, 0
        
        states_t = torch.cat(states, dim=0).to(self.device)
        logits, val_pred = self.model(states_t)
        
        targets = torch.zeros((len(states), 4096), device=self.device)
        for i, mp in enumerate(move_probs_list):
            for m, p in mp.items():
                idx = move_to_index(m)
                if 0 <= idx < 4096: targets[i, idx] = float(p)
                
        policy_loss = -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        value_loss = F.mse_loss(val_pred.view(-1), torch.FloatTensor(outcomes).to(self.device))
        loss = policy_loss + value_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return loss.item(), policy_loss.item(), value_loss.item()

    def save_model(self, path):
        torch.save({'model': self.model.state_dict(), 'optim': self.optimizer.state_dict()}, path)
        print(f"Saved to {path}")

    def load_model(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        try: self.optimizer.load_state_dict(ckpt['optim'])
        except: pass



class StockfishOpponent:
    def __init__(self, path=None, level=5):
        self.level = level
        self.path = path or shutil.which("stockfish")
        if not self.path: raise ValueError("Stockfish not found")
        self.engine = chess.engine.SimpleEngine.popen_uci(self.path)
        try: self.engine.configure({"Skill Level": level})
        except: pass
        print(f"Stockfish Lvl {level} ready")

    def get_move(self, board, limit=0.1):
        return self.engine.play(board, chess.engine.Limit(time=limit)).move

    def close(self):
        self.engine.quit()

class CurriculumTrainer:
    def __init__(self, agent, sf_path=None):
        self.agent = agent
        self.sf_path = sf_path

    def play_game(self, level, color, use_mcts):
        sf = StockfishOpponent(self.sf_path, level)
        board = chess.Board()
        data = {'s': [], 'p': [], 'v': []}
        
        try:
            while not board.is_game_over() and len(board.move_stack) < 400:
                if board.turn == color:
                    state = board_to_tensor(board)
                    if use_mcts: m, p, v = self.agent.select_move_mcts(board)
                    else: m, p, v = self.agent.select_move_fast(board)
                    if not m: break
                    data['s'].append(state)
                    data['p'].append(p)
                    data['v'].append(v)
                    board.push(m)
                else:
                    board.push(sf.get_move(board, 0.05))
        finally:
            sf.close()
            
        res = 1.0 if board.is_checkmate() and board.turn != color else -1.0 if board.is_checkmate() else 0.0
        return data['s'], data['p'], [res]*len(data['s']), res

    def train_curriculum(self, stages, games_per_stage=10, path="chess_bot_mcts.pth"):
        for stage, (lvl, desc) in enumerate(stages, 1):
            print(f"\nStage {stage}: {desc} (Lvl {lvl})")
            states, probs, outcomes = [], [], []
            w, l, d = 0, 0, 0
            
            for i in range(games_per_stage):
                color = chess.WHITE if i % 2 == 0 else chess.BLACK
                s, p, o, res = self.play_game(lvl, color, stage > 1)
                states.extend(s); probs.extend(p); outcomes.extend(o)
                if res > 0: w += 1; print(f"Game {i+1}: WIN")
                elif res < 0: l += 1; print(f"Game {i+1}: LOSS")
                else: d += 1; print(f"Game {i+1}: DRAW")

            if states:
                print(f"Training on {len(states)} positions...")
                bs = 32
                for i in range(0, len(states), bs):
                    self.agent.train_step(states[i:i+bs], probs[i:i+bs], None, outcomes[i:i+bs])
                self.agent.save_model(path)
                print(f"Stats: {w}W {d}D {l}L")



def print_board(board):
    print("\n  a b c d e f g h")
    print("  ---------------")
    for i in range(7, -1, -1):
        row = f"{i+1}|"
        for j in range(8):
            p = board.piece_at(i*8+j)
            row += (p.symbol() if p else ".") + " "
        print(f"{row}|{i+1}")
    print("  ---------------")
    print("  a b c d e f g h\n")

def get_human_move(board):
    while True:
        m_str = input("Your move (or 'quit'): ").strip()
        
        if m_str.lower() in ['quit', 'exit', 'resign']:
            return None
        
        try:
            move = chess.Move.from_uci(m_str)
            if move in board.legal_moves:
                return move
            else:
                print("Illegal move! Check piece movement or checks.")
        except:
            print("Invalid format! Use UCI (e.g., e2e4, a7a8q).")

def main():
    print("CHESS RL AGENT")
    print("1. Train (Quick)\n2. Train (Full)\n3. Play (MCTS)\n4. Play (Fast)")
    c = input("Choice: ").strip()
    
    agent = ChessAgent()
    sf_path = r"C:\PS\Semester---7\RL\Labsheets\Project\stockfish\stockfish-windows-x86-64-avx2.exe"
    
    if c in ['1', '2']:
        trainer = CurriculumTrainer(agent, sf_path)
        stages = [(1, "Basics")] if c == '1' else [(1, "Beginner"), (5, "Inter"), (10, "Advanced"), (12, "Expert")]
        games = 100 if c == '1' else 500
        trainer.train_curriculum(stages, games)
    elif c in ['3', '4']:
        if not os.path.exists("chess_bot_mcts.pth"): return print("Train first!")
        agent.load_model("chess_bot_mcts.pth")
        
        color_input = input("White/Black (w/b)? ").lower()
        my_color = chess.WHITE if color_input == 'w' else chess.BLACK
        board = chess.Board()
        
        print("\nGame Started! Type 'quit' to exit.")
        
        while not board.is_game_over():
            print_board(board)
            
            if board.turn == my_color:
                # Human Turn
                move = get_human_move(board)
                if move is None: 
                    print("Game Aborted.")
                    break
                board.push(move)
            else:
                # Bot Turn
                print("Bot thinking...")
                if c == '3':
                    move, _, v = agent.select_move_mcts(board)
                else:
                    move, _, v = agent.select_move_fast(board)
                
                if move is None:
                    print("Bot has no legal moves (Stalemate/Checkmate).")
                    break
                    
                print(f"Bot plays: {move.uci()} (Eval: {v:.2f})")
                board.push(move)
        
        print_board(board)
        print("Game Over")
        if board.is_checkmate():
            print(f"Winner: {'White' if not board.turn else 'Black'}")
        elif board.is_stalemate():
            print("Draw (Stalemate)")

if __name__ == "__main__":
    main()