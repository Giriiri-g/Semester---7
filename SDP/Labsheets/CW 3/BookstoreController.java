// BookstoreController.java

import java.util.Scanner;

public class BookstoreController {

    private BookstoreModel model;
    private BookstoreView view;

    public BookstoreController(BookstoreModel model, BookstoreView view) {
        this.model = model;
        this.view = view;
    }

    public void run() {
        Scanner sc = new Scanner(System.in);
        int choice;
        do {
            System.out.println("\n=== Online Bookstore ===");
            System.out.println("1. View Books");
            System.out.println("2. Add Book to Cart");
            System.out.println("3. View Cart");
            System.out.println("0. Exit");
            System.out.print("Enter choice: ");
            choice = sc.nextInt();
            switch (choice) {
                case 1:
                    view.showBooks(model.getBooks());
                    break;
                case 2:
                    System.out.print("Enter book ID to add: ");
                    int id = sc.nextInt();
                    Book book = model.getBookById(id);
                    if (book != null) {
                        model.addToCart(book);
                        view.showMessage(" " + book.getTitle() + " added to cart.");
                    } else {
                        view.showMessage(" Book not found.");
                    }
                    break;
                case 3:
                    view.showCart(model.getCart());
                    break;
                case 0:
                    view.showMessage(" Thanks for visiting the bookstore!");
                    break;
                default:
                    view.showMessage(" Invalid choice.");
            }
        } while (choice != 0);
        sc.close();
    }
}
