import socket
import sys

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

ips = socket.getaddrinfo("google.com", 80, proto=socket.IPPROTO_TCP)
server_address = ips[1][4]

print('connecting to %s port %s' % server_address)
sock.connect(server_address)

try:
    getRequest = ('GET / HTTP/1.1\r\n\r\n')
    print(getRequest)
    sock.sendall(getRequest.encode())

    amount_received = 0
    amount_expected = len(getRequest)

    while True:
        data = sock.recv(1024)
        amount_received += len(data)
        if amount_received == 0:
            break
        print('received "%s"' % data.decode())
finally:
    print('closing socket')
    sock.close()