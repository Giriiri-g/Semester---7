// Main.java

public class Main {

    public static void main(String[] args) {
        BookstoreModel model = new BookstoreModel();
        BookstoreView view = new BookstoreView();
        BookstoreController controller = new BookstoreController(model, view);
        controller.run();
    }
}
