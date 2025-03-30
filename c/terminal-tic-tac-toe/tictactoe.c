
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// Prints out a 3x3 2D array of chars.
void print_board(char board[3][3])
{
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            printf(" %c ", board[i][j]);

            if (j != 2)
            {
                printf("%c", '|');
            }
        }

        if (i != 2)
        {
            printf("\n-----------\n");
        }
        else
        {
            printf("\n");
        }
    }
}

// Checks the state of the tic tac toe game.
// Returns 0 if it is a tie
// Returns 1 if X is the winner
// Returns 2 if O is the winner
// Returns -1 if the game is still underway
int check_board_state(char board[3][3], int turns)
{
    // Check rows and columns
    for (int i = 0; i < 3; i++)
    {
        if (board[i][0] != ' ' && board[i][0] == board[i][1] && board[i][1] == board[i][2])
            return (board[i][0] == 'X') ? 1 : 2; // Row win

        if (board[0][i] != ' ' && board[0][i] == board[1][i] && board[1][i] == board[2][i])
            return (board[0][i] == 'X') ? 1 : 2; // Column win
    }

    // Check diagonals
    if (board[0][0] != ' ' && board[0][0] == board[1][1] && board[1][1] == board[2][2])
        return (board[0][0] == 'X') ? 1 : 2;

    if (board[0][2] != ' ' && board[0][2] == board[1][1] && board[1][1] == board[2][0])
        return (board[0][2] == 'X') ? 1 : 2;

    // Checking turn count to see if there are any more possible moves
    if (turns >= 9)
    {
        return 0;
    }
    else
    {
        return -1;
    }
}

// Runs a Tic Tac Toe game in the terminal.
int play_terminal_game()
{
    char board[3][3] = {};
    char default_value = ' ';

    // filling out the board.
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            board[i][j] = default_value;
        }
    }

    int turn_counter = 0;
    int board_state = -1;

    while (1)
    {
        // Start each iteration off by printing the board out.
        printf("\n===================================================\n\n");
        print_board(board);
        printf("\nPlayer %c's Turn:\n\n", turn_counter % 2 == 0 ? 'X' : 'O');

        size_t max_buffer_size = 8;
        char *input_buffer = (char *)malloc(max_buffer_size * sizeof(char));
        if (input_buffer == NULL)
        {
            printf("Failed to allocate memory for a message, quitting");
            exit(1);
        }

        // Reading / Handling User Input!
        fgets(input_buffer, max_buffer_size, stdin);

        if (input_buffer[0] == '\n') // if we got nothing, just return nothing
        {
            printf("Please provide a valid input");
            continue;
        }

        int x, y;
        if (strncmp(input_buffer, "quit", 4) == 0)
        {
            free(input_buffer);
            printf("\nGame quit by user.\n");
            exit(0);
        }

        // Parse input using sscanf
        if (sscanf(input_buffer, "%d,%d", &x, &y) != 2 || x < 0 || x > 2 || y < 0 || y > 2)
        {
            printf("Invalid, Please enter coordinates in 'x,y' format within (0-2,0-2)\n");
            free(input_buffer);
            continue;
        }

        // Place X or O
        if (board[x][y] == ' ')
        {
            board[x][y] = (turn_counter % 2 == 0) ? 'X' : 'O';
            ++turn_counter;
        }
        else
        {
            printf("Cell already occupied! Try again.\n");
            free(input_buffer);
            continue;
        }

        // Checking board state after user input is handled.
        free(input_buffer);
        board_state = check_board_state(board, turn_counter);

        if (board_state != -1)
        {
            break;
        }
    }

    printf("\n===========\n\n");
    print_board(board);
    return board_state;
}

// Main method to start execution of the program.
int main(void)
{
    printf("This is a terminal-based tic-tac-toe game program\n");
    printf("The \"X\" player will always go first\n");
    printf("Play by entering the coordinates as so: \"0,0\"\n");
    printf("(0,0) represents the top left, and (2,2) represents the bottom right\n");
    printf("You can also quit the game whenever by typing in \"quit\" instead\n");
    printf("Lets go!\n");

    int winner_code = play_terminal_game();

    if (winner_code == 1)
    {
        printf("\nPlayer X Won\n");
    }
    else if (winner_code == 2)
    {
        printf("\nPlayer O Won\n");
    }
    else
    {
        printf("\nReached a Tie\n");
    }

    return 0;
}
