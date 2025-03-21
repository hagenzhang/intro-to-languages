#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "reading.h"
#define BUFF_LENGTH 8
#define EXIT_WORD "quit"

char *read_next_line()
{
    // we allocate a default amount of space.
    // the amount allocated is small intentionally to ensure the code for
    // reallocating works cause i'm super inexperienced with memory allocation.

    // side note:
    // size_t is an unsigned integer type,
    // guaranteed to be large enough to
    // represent the size of any object in memory.
    // we can use it over int for:
    // - Array indexing
    // - Loop counters, especially when dealing with memory or data structures
    // - Representing the size of objects or memory regions
    size_t buffer_size = BUFF_LENGTH;

    // allocate memory
    char *msg_buffer = (char *)malloc(buffer_size * sizeof(char));

    // compare to null pointer to see if it failed to allocate memory
    if (msg_buffer == NULL)
    {
        printf("Failed to allocate memory for a message, quitting...");
        exit(1);
    }

    // now, we want to find the total size of the message the user puts in and
    // then realloc memory if necessary.
    while (1)
    {
        // read input with fgets, if it returns NULL we reached EOF!
        if (fgets(msg_buffer + strlen(msg_buffer), buffer_size - strlen(msg_buffer), stdin) == NULL)
        {
            if (strlen(msg_buffer) == 0) // if we got nothing, just return nothing
            {
                free(msg_buffer);
                return NULL;
            }
            return msg_buffer; // otherwise, return our buffer
        }


        // if we are here, then we haven't finished reading yet (buffer too small?).
        // check if last character is newline.
        // we need to do this to make sure when the user hits "enter" it gets interpreted
        // as the end of the message.
        size_t len = strlen(msg_buffer);
        if (len > 0 && msg_buffer[len - 1] == '\n')
        {
            msg_buffer[len - 1] = '\0'; // replace newline with null terminator
            return msg_buffer;
        }

        // now, resize buffer since we couldn't read the entire thing.
        buffer_size += BUFF_LENGTH;
        char *temp = (char *)realloc(msg_buffer, buffer_size * sizeof(char));
        if (!temp)
        {
            free(msg_buffer);
            printf("Memory reallocation failed, quitting...\n");
            return NULL;
        }
        msg_buffer = temp;
    }
}

int main()
{
    // hello messages
    printf("Hello, user!\n");
    printf("Please enter a string to hear it parroted, or type \"quit\" to exit\n");

    char *message;

    while (1)
    {
        message = read_next_line();

        if (!message || strcmp(message, EXIT_WORD) == 0)
        {
            free(message);
            break;
        }

        printf("You entered: %s\n", message);
        free(message);
    }

    return 0;
}