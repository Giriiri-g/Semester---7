import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class DecoratorPatternCustomer {

    public static void main(String args[]) {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        int choice = 0;

        while (true) {
            System.out.print("========= Food Menu ============ \n");
            System.out.print("            1. Vegetarian Food.   \n");
            System.out.print("            2. Non-Vegetarian Food.\n");
            System.out.print("            3. Chineese Food.         \n");
            System.out.print("            4. Exit                        \n");
            System.out.print("Enter your choice: ");

            try {
                String line = br.readLine();
                if (line == null) { // EOF protection
                    System.out.println("No input. Exiting.");
                    break;
                }
                choice = Integer.parseInt(line.trim());
            } catch (NumberFormatException | IOException e) {
                System.out.println("Invalid input. Please enter a number between 1 and 4.");
                continue;
            }

            switch (choice) {
                case 1: {
                    VegFood vf = new VegFood();
                    System.out.println(vf.prepareFood());
                    System.out.println(vf.foodPrice());
                }
                break;

                case 2: {
                    Food f1 = new NonVegFood(new VegFood());
                    System.out.println(f1.prepareFood());
                    System.out.println(f1.foodPrice());
                }
                break;

                case 3: {
                    Food f2 = new ChineeseFood(new VegFood());
                    System.out.println(f2.prepareFood());
                    System.out.println(f2.foodPrice());
                }
                break;

                case 4: {
                    System.out.println("Exiting. Thank you!");
                    return; // exit program
                }

                default: {
                    System.out.println("Other than these no food available");
                }
                break;
            }
        }
    }
}