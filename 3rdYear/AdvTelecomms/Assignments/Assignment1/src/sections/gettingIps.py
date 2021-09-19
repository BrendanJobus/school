import socket

ips = socket.getaddrinfo("google.com", 80, proto=socket.IPPROTO_TCP)

for ip in ips:
    print(ip[4][0])