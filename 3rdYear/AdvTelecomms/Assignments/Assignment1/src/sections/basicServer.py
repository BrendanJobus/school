import socket
import sys

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_address = ('localhost', 10000)
print('starting up on %s port %s' % server_address)
sock.bind(server_address)

sock.listen(1)

while True:
    print('waiting for a connection')
    connection, client_address = sock.accept()
    try:
        print('connection from %s %s' % client_address)
        while True:
            data = connection.recv(8)
            print('received "%s"' % data.decode())
            if data:
                print('sending data back to the client')
                connection.sendall(data)
            else:
                print('no more data from %s %s' % client_address)
                break
    finally:
        connection.close()