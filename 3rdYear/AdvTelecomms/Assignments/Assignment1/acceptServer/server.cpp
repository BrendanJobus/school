#include <cstdlib>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <stdlib.h>
#include <unistd.h>
#include <netdb.h>

#define ACCEPT_PORT 9999

int main() {
    int serverfd, clientfd, opt = 1;

    if ((serverfd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        perror("Error: failed to create socket\n");
        exit(EXIT_FAILURE);
    }

    if (setsockopt(serverfd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
        perror("Error: failed to set socket options");
        exit(EXIT_FAILURE);
    }

    struct sockaddr_in server, client;
    socklen_t clilen = sizeof(client);
    int port = ACCEPT_PORT;

    bzero((char *) &server, sizeof(server));
    server.sin_family = AF_INET;
    server.sin_addr.s_addr = INADDR_ANY;
    server.sin_port = htons(port);
    if (bind(serverfd, (struct sockaddr *) &server, sizeof(server)) < 0) {
        perror("Error: failed to bind\n");
        exit(EXIT_FAILURE);
    }

    listen(serverfd, 5);

    clientfd = accept(serverfd, (struct sockaddr *) &client, &clilen);
    if (clientfd < 0) {
        perror("Error: failed to accept");
        exit(EXIT_FAILURE);
    }

    printf("server: got connection from %s port %d\n", inet_ntoa(client.sin_addr), ntohs(client.sin_port));
}