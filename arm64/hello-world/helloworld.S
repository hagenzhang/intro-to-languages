.data
message:
    .ascii "Hello, World!\n"

.text
.global _main          // entry point

_main:
    mov x0, #1          // write to stdout
    ldr x1, =message    // pass string to x1, using LDR rather than move bc it's memory
    mov x2, #14         // moving length of message to x2
    mov w8, #64         // passing in write syscall value
    svc #0              // making the actual syscall
    b .                 // infinite looping 


// run via:
// as helloworld.S -o helloworld.o -> gnu assembler
// ld helloworld.o -> running linker

