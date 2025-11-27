
import java.util.*;

public class BookstoreModel {

    private List<Book> books = new ArrayList<>();
    private List<Book> cart = new ArrayList<>();

    public BookstoreModel() {
        // Sample data
        books.add(new Book(1, "Clean Code", "Robert C. Martin", 30.0));
        books.add(new Book(2, "Design Patterns", "GoF", 40.0));
        books.add(new Book(3, "Effective Java", "Joshua Bloch", 35.0));
    }

    public List<Book> getBooks() {
        return books;
    }

    public List<Book> getCart() {
        return cart;
    }

    public Book getBookById(int id) {
        for (Book b : books) {
            if (b.getId() == id) {
                return b;
            }
        }
        return null;
    }

    public void addToCart(Book book) {
        if (book != null) {
            cart.add(book);
        }
    }
}
