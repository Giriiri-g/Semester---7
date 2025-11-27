# chess_mcts_agent.py
# Chess Deep RL Bot with MCTS - Fixed & complete script

import os
import math
import time
import shutil
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import chess
import chess.engine

# ===================== Neural Network Model =====================

class ResidualBlock(nn.Module):
    """Residual block for deeper network"""
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
    """Enhanced neural network with residual blocks"""
    
    def __init__(self):
        super(ChessNet, self).__init__()
        
        # Input: 14 channels (12 pieces + 2 meta channels)
        self.conv_input = nn.Conv2d(14, 128, 3, padding=1)
        self.bn_input = nn.BatchNorm2d(128)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([ResidualBlock(128) for _ in range(6)])
        
        # Policy head
        self.policy_conv = nn.Conv2d(128, 32, 1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, 4096)
        
        # Value head
        self.value_conv = nn.Conv2d(128, 16, 1)
        self.value_bn = nn.BatchNorm2d(16)
        self.value_fc1 = nn.Linear(16 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
    def forward(self, x):
        # Input processing
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # Residual blocks
        for block in self.res_blocks:
            x = block(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 32 * 8 * 8)
        policy = self.policy_fc(policy)  # shape (B, 4096)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 16 * 8 * 8)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))  # shape (B,1)
        
        return policy, value


# ===================== Board Encoding =====================

def board_to_tensor(board):
    """Convert chess board to neural network input with meta information"""
    # 14 channels: 12 pieces + kingside_castle + queenside_castle + en-passant mapped into channel 13
    # We'll use channel indices:
    # 0-5: white pawn..king, 6-11: black pawn..king, 12: castling plane(s), 13: en-passant (one-hot square)
    tensor = np.zeros((14, 8, 8), dtype=np.float32)
    
    piece_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    
    # Piece positions
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # square -> (row, col) where row 0 is rank1 (a1) and col 0 is file a
            row, col = divmod(square, 8)
            channel = piece_map[piece.piece_type]
            if not piece.color:  # Black
                channel += 6
            tensor[channel, row, col] = 1.0
    
    # Castling: use plane 12 as a global indicator for the side to move's castling rights
    # Encode kingside as +1.0, queenside as +0.5 (keeps one plane but distinct values)
    # (You could separate to two planes for clarity; this keeps your original spirit.)
    castling_plane_value = 0.0
    if board.has_kingside_castling_rights(board.turn):
        castling_plane_value += 1.0
    if board.has_queenside_castling_rights(board.turn):
        castling_plane_value += 0.5
    tensor[12, :, :] = castling_plane_value  # fill entire plane with same scalar
    
    # En-passant: one-hot in channel 13 if present
    if board.ep_square is not None:
        r, c = divmod(board.ep_square, 8)
        tensor[13, r, c] = 1.0
    
    return torch.FloatTensor(tensor).unsqueeze(0)  # shape (1,14,8,8)


def move_to_index(move):
    """Convert chess.Move to index in [0,4095]"""
    return int(move.from_square) * 64 + int(move.to_square)


def index_to_move(index, board):
    """Convert index back to move if legal on board, else try promotions"""
    from_sq = index // 64
    to_sq = index % 64
    move = chess.Move(from_sq, to_sq)
    
    if move in board.legal_moves:
        return move
    for promotion in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
        promo_move = chess.Move(from_sq, to_sq, promotion=promotion)
        if promo_move in board.legal_moves:
            return promo_move
    return None


# ===================== MCTS Implementation =====================

class MCTSNode:
    """Node in MCTS tree"""
    def __init__(self, board, parent=None, move=None, prior=0.0):
        self.board = board.copy()
        self.parent = parent
        self.move = move
        self.prior = float(prior)
        
        self.children = {}          # move -> MCTSNode
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False
        
    def value(self):
        """Average value of this node"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, c_puct=1.5):
        """UCB score for node selection"""
        if self.parent is None:
            return float('inf')
        parent_visits = max(1, self.parent.visit_count)
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.value() + exploration
    
    def select_child(self):
        """Select child with highest UCB score"""
        return max(self.children.values(), key=lambda n: n.ucb_score())
    
    def expand(self, policy_probs):
        """Expand node with legal moves and priors from policy_probs (length 4096)"""
        if self.is_expanded:
            return
        
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            self.is_expanded = True
            return
        
        for move in legal_moves:
            move_idx = move_to_index(move)
            prior = float(policy_probs[move_idx]) if (0 <= move_idx < len(policy_probs)) else 1e-6
            new_board = self.board.copy()
            new_board.push(move)
            self.children[move] = MCTSNode(new_board, parent=self, move=move, prior=prior)
        
        self.is_expanded = True
    
    def backup(self, value):
        """Backpropagate value up the tree recursively. value is from this node's perspective."""
        self.visit_count += 1
        self.value_sum += value
        if self.parent:
            # flip sign for parent (opponent perspective)
            self.parent.backup(-value)


class MCTS:
    """Monte Carlo Tree Search"""
    def __init__(self, agent, num_simulations=200):
        self.agent = agent
        self.num_simulations = num_simulations
    
    def search(self, board):
        """Perform MCTS and return (move_probs dict, root_value)."""
        root = MCTSNode(board)
        
        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            
            # selection: traverse to leaf
            while node.is_expanded and node.children:
                node = node.select_child()
            
            # expansion & evaluation
            if not node.board.is_game_over():
                # Evaluate with neural net
                # ensure model in eval mode and no grad
                self.agent.model.eval()
                state = board_to_tensor(node.board).to(self.agent.device)
                with torch.no_grad():
                    policy_logits, value = self.agent.model(state)  # policy_logits shape (1,4096)
                    policy_probs = F.softmax(policy_logits[0], dim=0).cpu().numpy()
                    value = float(value.item())
                
                node.expand(policy_probs)
            else:
                # terminal node
                if node.board.is_checkmate():
                    # player to move lost -> value = -1 for the player to move at this node
                    value = -1.0
                else:
                    value = 0.0
            
            # backup once (recursive inside)
            node.backup(value)
        
        # Build visit-count normalized probabilities for root children
        move_probs = {}
        total_visits = sum(child.visit_count for child in root.children.values())
        total_visits = max(1, total_visits)
        for move, child in root.children.items():
            move_probs[move] = child.visit_count / total_visits
        
        return move_probs, root.value()


# ===================== Enhanced RL Agent =====================

class ChessAgent:
    """Deep RL Chess Agent with MCTS"""
    def __init__(self, model_path=None, device=None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChessNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.mcts_simulations = 100  # default; adjustable
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            print(f"✓ Loaded model from {model_path}")
    
    def select_move_mcts(self, board, temperature=1.0):
        """Select move using MCTS"""
        mcts = MCTS(self, num_simulations=self.mcts_simulations)
        move_probs, value = mcts.search(board)
        
        if not move_probs:
            return None, {}, 0.0
        
        moves = list(move_probs.keys())
        probs = np.array([move_probs[m] for m in moves], dtype=np.float64)
        
        if temperature < 0.1:
            # greedy pick
            idx = int(np.argmax(probs))
            move = moves[idx]
        else:
            probs = probs ** (1.0 / temperature)
            probs_sum = probs.sum()
            if probs_sum <= 0:
                probs = np.ones_like(probs) / len(probs)
            else:
                probs = probs / probs_sum
            move = np.random.choice(moves, p=probs)
        
        return move, move_probs, value
    
    def select_move_fast(self, board, temperature=1.0):
        """Fast move selection without MCTS (for training speed)"""
        self.model.eval()
        with torch.no_grad():
            state = board_to_tensor(board).to(self.device)
            policy_logits, value = self.model(state)  # policy_logits shape (1,4096)
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None, {}, 0.0
        
        legal_indices = [move_to_index(m) for m in legal_moves]
        logits = policy_logits[0, legal_indices] / max(1e-8, temperature)
        probs = F.softmax(logits, dim=0).cpu()
        
        # sample
        idx = torch.multinomial(probs, 1).item()
        selected_move = legal_moves[idx]
        
        move_dict = {m: float(probs[i].item()) for i, m in enumerate(legal_moves)}
        return selected_move, move_dict, float(value.item())
    
    def train_step(self, states, move_probs_list, values, outcomes):
        """Train on collected experience (states: list of tensors shape (1,14,8,8))"""
        self.model.train()
        batch_size = len(states)
        if batch_size == 0:
            return 0.0, 0.0, 0.0
        
        states_tensor = torch.cat(states, dim=0).to(self.device)  # shape (B,14,8,8)
        
        # Forward
        policy_logits, values_pred = self.model(states_tensor)  # shapes (B,4096), (B,1)
        values_pred = values_pred.view(-1)  # shape (B,)
        
        # Build dense targets for policy (B,4096)
        targets = torch.zeros((batch_size, 4096), dtype=torch.float32, device=self.device)
        for i, move_probs in enumerate(move_probs_list):
            # move_probs is a dict move -> prob
            for move, prob in move_probs.items():
                idx = move_to_index(move)
                if 0 <= idx < 4096:
                    targets[i, idx] = float(prob)
        
        log_probs = F.log_softmax(policy_logits, dim=1)
        policy_loss = - (targets * log_probs).sum(dim=1).mean()
        
        outcomes_tensor = torch.FloatTensor(outcomes).to(self.device)
        value_loss = F.mse_loss(values_pred, outcomes_tensor)
        
        loss = policy_loss + value_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return float(loss.item()), float(policy_loss.item()), float(value_loss.item())
    
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"✓ Model saved to {path}")
    
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception:
            # If optimizer state doesn't match (e.g., architecture changed), skip loading optimizer
            pass


# ===================== Stockfish Opponent =====================

class StockfishOpponent:
    """Play against Stockfish engine at various skill levels"""
    def __init__(self, stockfish_path=None, skill_level=5):
        """
        skill_level: 0-20 (0=beginner, 20=master)
        """
        self.skill_level = int(skill_level)
        
        # Try to find Stockfish
        if stockfish_path is None:
            stockfish_path = self._find_stockfish()
        
        if stockfish_path is None:
            raise ValueError(
                "Stockfish not found! Please:\n"
                "1. Install Stockfish and ensure it's on your PATH, or\n"
                "2. Provide the engine path when prompted.\n"
                "Download: https://stockfishchess.org/download/"
            )
        
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            # Not all builds use the same option name; try to set skill if available
            try:
                self.engine.configure({"Skill Level": self.skill_level})
            except Exception:
                # ignore if engine doesn't support this option
                pass
            print(f"✓ Stockfish loaded (Skill Level: {self.skill_level})")
        except Exception as e:
            raise ValueError(f"Failed to load Stockfish from {stockfish_path}: {e}")
    
    def _find_stockfish(self):
        # prefer PATH executable
        exe = shutil.which("stockfish")
        if exe:
            return exe
        # common locations
        common_paths = [
            "/usr/local/bin/stockfish",
            "/usr/bin/stockfish",
            "/opt/homebrew/bin/stockfish",
            "C:\\Program Files\\Stockfish\\stockfish.exe",
            "stockfish.exe"
        ]
        for p in common_paths:
            if os.path.exists(p) and os.access(p, os.X_OK):
                return p
        return None
    
    def get_move(self, board, time_limit=0.1):
        """Get move from Stockfish with a small time limit (seconds)"""
        result = self.engine.play(board, chess.engine.Limit(time=time_limit))
        return result.move
    
    def close(self):
        try:
            self.engine.quit()
        except Exception:
            pass


# ===================== Training Against Stockfish =====================

class CurriculumTrainer:
    """Train agent against Stockfish with curriculum learning"""
    def __init__(self, agent, stockfish_path=None):
        self.agent = agent
        self.stockfish_path = stockfish_path
        self.game_history = []
    
    def play_vs_stockfish(self, skill_level=5, agent_color=chess.WHITE, use_mcts=False):
        """Play one game against Stockfish and collect training positions."""
        stockfish = StockfishOpponent(self.stockfish_path, skill_level)
        board = chess.Board()
        
        states, move_probs_list, values = [], [], []
        
        try:
            move_count = 0
            while not board.is_game_over() and move_count < 400:
                if board.turn == agent_color:
                    state = board_to_tensor(board)
                    if use_mcts:
                        move, move_probs, value = self.agent.select_move_mcts(board, temperature=1.0)
                    else:
                        move, move_probs, value = self.agent.select_move_fast(board, temperature=1.0)
                    
                    if move is None:
                        break
                    
                    states.append(state)
                    move_probs_list.append(move_probs)
                    values.append(value)
                    board.push(move)
                else:
                    move = stockfish.get_move(board, time_limit=0.05)
                    board.push(move)
                move_count += 1
        finally:
            stockfish.close()
        
        # Outcome from agent perspective
        if board.is_checkmate():
            winner = not board.turn
            outcome = 1.0 if winner == agent_color else -1.0
        else:
            outcome = 0.0
        
        outcomes = [outcome] * len(states)
        return states, move_probs_list, outcomes, outcome
    
    def train_curriculum(self, stages, games_per_stage=10, model_path="chess_bot_mcts.pth"):
        print("\n" + "="*60)
        print("CURRICULUM TRAINING AGAINST STOCKFISH")
        print("="*60)
        
        for stage_num, (skill_level, description) in enumerate(stages, 1):
            print(f"\n{'='*60}")
            print(f"Stage {stage_num}/{len(stages)}: {description}")
            print(f"Stockfish Skill Level: {skill_level}")
            print(f"Games: {games_per_stage}")
            print('='*60)
            
            wins, losses, draws = 0, 0, 0
            all_states, all_move_probs, all_outcomes = [], [], []
            
            for game_num in range(games_per_stage):
                agent_color = chess.WHITE if game_num % 2 == 0 else chess.BLACK
                color_str = "White" if agent_color == chess.WHITE else "Black"
                print(f"  Game {game_num+1}/{games_per_stage} (Agent: {color_str})...", end=" ")
                
                states, move_probs, outcomes, result = self.play_vs_stockfish(
                    skill_level, agent_color, use_mcts=(stage_num > 1)
                )
                all_states.extend(states)
                all_move_probs.extend(move_probs)
                all_outcomes.extend(outcomes)
                
                if result > 0:
                    wins += 1
                    print("WIN ✓")
                elif result < 0:
                    losses += 1
                    print("LOSS ✗")
                else:
                    draws += 1
                    print("DRAW -")
            
            # Train on collected games if any
            if all_states:
                print(f"\n  Training on {len(all_states)} positions...")
                batch_size = 32
                num_batches = math.ceil(len(all_states) / batch_size)
                total_loss = 0.0
                for i in range(num_batches):
                    start = i * batch_size
                    end = min(start + batch_size, len(all_states))
                    batch_states = all_states[start:end]
                    batch_move_probs = all_move_probs[start:end]
                    batch_outcomes = all_outcomes[start:end]
                    
                    loss, policy_loss, value_loss = self.agent.train_step(
                        batch_states, batch_move_probs, None, batch_outcomes
                    )
                    total_loss += loss
                avg_loss = total_loss / max(1, num_batches)
                print(f"  Results: {wins}W / {draws}D / {losses}L")
                print(f"  Win Rate: {wins/games_per_stage*100:.1f}%")
                print(f"  Avg Loss: {avg_loss:.4f}")
            else:
                print("  No positions collected this stage.")
            
            self.agent.save_model(model_path)
        
        print("\n" + "="*60)
        print("CURRICULUM TRAINING COMPLETE!")
        print("="*60)


# ===================== Game Interface =====================

def print_board(board):
    """Print chess board to terminal"""
    print("\n  a b c d e f g h")
    print("  ---------------")
    for i in range(7, -1, -1):
        row = f"{i+1}|"
        for j in range(8):
            piece = board.piece_at(i * 8 + j)
            if piece:
                row += piece.symbol() + " "
            else:
                row += ". "
        row += f"|{i+1}"
        print(row)
    print("  ---------------")
    print("  a b c d e f g h\n")


def get_human_move(board):
    """Get move from human player"""
    while True:
        try:
            move_str = input("Your move (e.g., e2e4): ").strip()
            if move_str.lower() in ['quit', 'exit', 'resign']:
                return None
            move = chess.Move.from_uci(move_str)
            if move in board.legal_moves:
                return move
            else:
                print("Illegal move! Try again.")
        except Exception:
            print("Invalid format! Use UCI notation (e.g., e2e4)")


def play_against_bot(agent, human_color=chess.WHITE, use_mcts=True):
    """Play a game against the bot"""
    board = chess.Board()
    print("\n" + "="*60)
    print("CHESS DEEP RL BOT WITH MCTS")
    print("="*60)
    print(f"You: {'White' if human_color else 'Black'}")
    print(f"Bot: {'Black' if human_color else 'White'} (MCTS: {'ON' if use_mcts else 'OFF'})")
    print("="*60)
    
    while not board.is_game_over():
        print_board(board)
        print(f"Move {board.fullmove_number} - {'White' if board.turn else 'Black'} to move")
        
        if board.turn == human_color:
            move = get_human_move(board)
            if move is None:
                print("You resigned!")
                break
        else:
            print("Bot thinking...", end=" ", flush=True)
            start = time.time()
            if use_mcts:
                move, move_probs, value = agent.select_move_mcts(board, temperature=0.1)
            else:
                move, move_probs, value = agent.select_move_fast(board, temperature=0.1)
            elapsed = time.time() - start
            if move is None:
                print("\nBot has no legal moves.")
                break
            print(f"\nBot plays: {move.uci()} (eval: {value:.2f}, time: {elapsed:.2f}s)")
        
        board.push(move)
    
    print_board(board)
    print("\n" + "="*60)
    print("GAME OVER!")
    if board.is_checkmate():
        winner = "White" if not board.turn else "Black"
        print(f"Checkmate! {winner} wins!")
    elif board.is_stalemate():
        print("Stalemate - Draw")
    else:
        print("Draw")
    print("="*60)


# ===================== Main Entry Point =====================

def main():
    print("\n" + "="*60)
    print("CHESS DEEP RL BOT WITH MCTS + STOCKFISH TRAINING")
    print("="*60)
    print("\nSelect mode:")
    print("1. Quick train (5 games vs Stockfish Lvl 1-5)")
    print("2. Full curriculum (30 games vs Stockfish Lvl 1-10)")
    print("3. Advanced curriculum (60 games vs Stockfish Lvl 1-15)")
    print("4. Play vs trained bot (with MCTS)")
    print("5. Play vs trained bot (fast mode, no MCTS)")
    
    choice = input("\nChoice (1-5): ").strip()
    if choice in ["1", "2", "3"]:
        agent = ChessAgent()
        stockfish_path = input("\nStockfish path (press Enter for auto-detect): ").strip()
        if not stockfish_path:
            stockfish_path = None
        trainer = CurriculumTrainer(agent, stockfish_path)
        if choice == "1":
            stages = [(1, "Beginner - Learning basics"), (3, "Novice - Basic tactics"), (5, "Intermediate - Strategic play")]
            trainer.train_curriculum(stages, games_per_stage=5)
        elif choice == "2":
            stages = [(1, "Beginner"), (3, "Novice"), (5, "Intermediate"), (7, "Advanced"), (10, "Expert")]
            trainer.train_curriculum(stages, games_per_stage=6)
        elif choice == "3":
            stages = [(1, "Level 1 - Beginner"), (3, "Level 3 - Novice"), (5, "Level 5 - Intermediate"),
                      (8, "Level 8 - Advanced"), (12, "Level 12 - Expert"), (15, "Level 15 - Master")]
            trainer.train_curriculum(stages, games_per_stage=30)
        print("\nTraining complete! Now you can play (option 4 or 5)")
    elif choice in ["4", "5"]:
        model_path = "chess_bot_mcts.pth"
        if not os.path.exists(model_path):
            print(f"\n✗ Model not found: {model_path}")
            print("Please train first (options 1-3)")
            return
        agent = ChessAgent(model_path)
        color_choice = input("\nPlay as White or Black? (w/b): ").strip().lower()
        human_color = chess.WHITE if color_choice == "w" else chess.BLACK
        use_mcts = (choice == "4")
        if use_mcts:
            sims = input("MCTS simulations (default 100, higher=stronger): ").strip()
            agent.mcts_simulations = int(sims) if sims else 100
        play_against_bot(agent, human_color, use_mcts)
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()
