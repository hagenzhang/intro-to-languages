#ifndef READING_H
#define READING_H
/*
Above is a "guard" for c header files.
It ensures that the header file is only defined once to avoid the problem 
of double inclusion when adding include dependencies.

According to Wikipedia:

"... However, if an #include directive for a given file appears multiple times during
compilation, the code will effectively be duplicated in that file. If the included
file includes a definition, this can cause a compilation error due to the
One Definition Rule, which says that definitions (such as the definition of a class)
cannot be duplicated in a translation unit. #include guards prevent this by defining
a preprocessor macro when a header is first included. In the event that header file
is included a second time, the #include guard will prevent the actual code within
that header from being compiled."

https://en.wikipedia.org/wiki/Include_guard


Header files are used when you want to declare functions, classes, or other symbols
that will be used across multiple source files. While it isn't necessary for this
project, I added one anyways just so I could toy around with it.
*/

// function to read the next line from stdin
char* read_next_line();

#endif // End of the include guard