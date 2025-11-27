
import java.util.List;

public class BookstoreView {

    public void showBooks(List<Book> books) {
        System.out.println("=== Available Books ===");
        for (Book b : books) {
            System.out.println(b.getId() + ". " + b.getTitle() + " by " + b.getAuthor() + " - $"
                    + b.getPrice());
        }
    }

    public void showCart(List<Book> cart) {
        System.out.println("=== Your Cart ===");
        if (cart.isEmpty()) {
            System.out.println("Cart is empty.");
        } else {
            double total = 0;
            for (Book b : cart) {
                System.out.println(b.getTitle() + " - $" + b.getPrice());
                total += b.getPrice();
            }
            System.out.println("Total: $" + total);
        }
    }

    public void showMessage(String message) {
        System.out.println(message);
    }
}
