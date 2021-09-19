import email
import logging
import socket
import ssl
import string
import threading
import time
from io import StringIO



### Things TODO: create a way to implement https and then have a way to choose between http and https depending on what the client wants, implement multithreading

def createSocket():
    ### Creating and setting up the socket ###
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR | socket.SO_REUSEPORT, 1)
    return server_sock

def bindAndListen(server_sock, port):
    ### Binding the socket ###
    server_address = ('localhost', port)
    print('starting up on %s port %s' % server_address)
    server_sock.bind(server_address)

    ### Setting the socket to listen with backlog of 5 ###
    server_sock.listen(5)

def requestLoop(client_sock):
    request = ""
    while True:
        data = client_sock.recv(1024)

        ### We have two variables that are receiving decoded data, we need two so that we can search on for an end of http request and one to store the entire request ###
        decodedData = data.decode()
        request += decodedData
        if "\r\n\r\n" in decodedData:
            break
    return request

def splitAndFormatRequest(request):
    ### Here, we are splitting the request into headers and the main request so that we can utilize the data to create a new GET request that we can use ###
    getRequest, headers = request.split('\r\n', 1)
    deconstructed = email.message_from_file(StringIO(headers))
    headers = dict(deconstructed.items())

    ### Creating a new GET request for the server ###
    getRequest += "\r\n\r\n"

    ### removing the host from the GET request as it will cause an error ###
    removeHost = "http://"
    removeHost += headers['Host']
    newRequest = getRequest.replace(removeHost, '')

    return newRequest, headers

def getFirstAddress(headers):
    ### were using getaddrinfo to get the ip address of the domain that is stored in socket ###
    host = ""
    port = ""
    try:
        host, port = headers['Host'].split(':', 1)
        headers['Host'] = host
        ips = socket.getaddrinfo(host, int(port), proto=socket.IPPROTO_TCP)
    except ValueError:
        ips = socket.getaddrinfo(headers['Host'], 80, proto=socket.IPPROTO_TCP)

    ### we use the first address to connect to the server ###
    web_address = ips[0][4]
    return web_address

def acceptLoop(web_sock, client_sock):
    try:
        toSendBack = bytes()
        web_sock.settimeout(0.5)

        ### Loop for accepting data from server ###
        while True:
            data = web_sock.recv(1024)
            toSendBack += data
            if len(data) == 0:
                break
        print(toSendBack.decode())
        client_sock.sendall(toSendBack)
    except socket.timeout:
        client_sock.sendall(toSendBack)
    finally:
        web_sock.settimeout(0)

def main():
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format = format, level = logging.INFO, datefmt = "%H:%M:%S %d/%m/%y")

    socket.ssl
    portno = 10000
    numberOfRequests = 0
    server_sock = createSocket()
    bindAndListen(server_sock, portno)

    context = ssl.create_default_context()
    context.load_verify_locations('cert.pem')

    logging.info("Waiting for a connection\n")

    ### This our main loop, it will keep our proxy running and accepting new requests ###
    while True:
        ### Accepting a connection and making the new socket connection, this is the client that we are connected to ###
        client_sock, client_address = server_sock.accept()

        ### creating a web_socket, the socket that the we will use to connect to the server has connected through ###
        web_sock = createSocket()

        try:
            logging.info('Connection from %s %s\n' % client_address)
            request = requestLoop(client_sock)
            numberOfRequests += 1
            requestNo = numberOfRequests

            logging.info("Received request: id %i \n%s" % (requestNo, request))
            getRequest, headers = splitAndFormatRequest(request)
            web_address = getFirstAddress(headers)

            print("%s" % getRequest)
            
            web_sock.connect(web_address)
            secure_web_sock = context.wrap_socket(web_sock, server_hostname=headers['Host'])

            secure_web_sock.sendall("CONNECT www.google.com:443 HTTP/1.1\r\nHost: google.com\r\n\r\n".encode())
            #secure_web_sock.sendall(getRequest.encode())

            #connectRequest = "CONNECT google.com:443 HTTP/1.1\r\n\r\n"
            #secure_web_sock.sendall(connectRequest.encode())

            #web_sock.connect(web_address)
            #web_sock.sendall(getRequest.encode())

            acceptLoop(secure_web_sock, client_sock)
            logging.info("Complete request on id %i\n" % requestNo)
        #except socket.gaierror:
            #logging.info("Request failed on id %i\n" % requestNo)
        finally:
            client_sock.close()
            web_sock.close()

    server_sock.close()

if __name__ == "__main__":
    main()