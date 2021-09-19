#include <string>

int createReusableSocket(int domain, int type, int option);

void createBind(int fd, struct sockaddr_in *server, int port);

void acceptHandler();

int connectToServer(char *szServerName, int port);

void mParseUrl(char *mUrl, std::string &serverName, std::string &filepath, std::string &filename);

int getHeaderLength(char *content);