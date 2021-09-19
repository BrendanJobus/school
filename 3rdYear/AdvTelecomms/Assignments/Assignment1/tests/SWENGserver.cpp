#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h> 
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <thread>
#include <iostream>

void error(const char *msg) {
    perror(msg);
    exit(1);
}

void acceptHandler(int clientSockFd, struct sockaddr_in cli_addr) {
    char buffer[256];
    int n;

    printf("server: got connection from %s port %d\n", inet_ntoa(cli_addr.sin_addr), ntohs(cli_addr.sin_port));
    send(clientSockFd, "Hello, world!\n", 13, 0);

    bzero(buffer, 256);
    n = read(clientSockFd, buffer, 255);
    if (n < 0) error("ERROR reading from socket");
        printf("Here is the message: %s\n", buffer);
    close(clientSockFd);
}

void acceptFunc(int sockfd, struct sockaddr_in cli_addr) {
    while(1) {
        socklen_t clilen = sizeof(cli_addr);
        int newsockfd = accept(sockfd, (struct sockaddr *) &cli_addr, &clilen);
        if (newsockfd < 0) error("ERROR on accept");
        acceptHandler(newsockfd, cli_addr);
    }
}

int main(int argc, char *argv[]) {
    int sockfd, newsockfd, portno, sockbinding, socket_data, opt = 1;

    socklen_t clilen;
    char buffer[255];

    struct sockaddr_in serv_addr, cli_addr;
    int n;

    if (argc < 2) {
        fprintf(stderr, "ERROR, no port provided\n");
        exit(1);
    }

    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) error("ERROR opening socket");

    if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }

    bzero((char *) &serv_addr, sizeof(serv_addr));
    portno = atoi(argv[1]);

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = htons(portno);

    if (bind(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0)
        error("ERROR on binding");

    listen(sockfd, 5);

    clilen = sizeof(cli_addr);

    // This will start an infinite loop, however, we will stop at accept every time, this is because accept is a blocking function
    // TODO: make some sort of function that will be called every time accept occurs, this function will start a new thread that will then handle that connection
/*    while(1) {
        newsockfd = accept(sockfd, (struct sockaddr *) &cli_addr, &clilen);
        if (newsockfd < 0) error("ERROR on accept");
        std::thread t(acceptHandler, newsockfd, cli_addr);
        t.join();
    }
*/

    // maybe instead of a single while loop, we create multiple threads, each one dealling
    // with a specific task, and they will instead have a while loop in them

    std::thread t(acceptFunc, sockfd, cli_addr);

    t.join();

    close(sockfd);
    return 0;
}