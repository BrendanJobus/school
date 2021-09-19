import email
import logging
import os
import socket
import ssl
import string
import threading
import time
from cmd import Cmd
from datetime import datetime, timedelta
from functools import lru_cache, wraps
from io import StringIO, BytesIO

### Things TODO: create a way to implement https and then have a way to choose between http and https depending on what the client wants, implement multithreading

class console(Cmd):
    prompt = 'Proxy> '
    intro = "The Proxy has been started! Type ? for help"
    blacklist_file = ""

    def do_exit(self, inp):
        writeBlacklist(self.blacklist_file)
        print("Goodbye")
        return True

    def help_exit(self):
        print("exit the application. Shorthand: x q Ctrl-D.")

    def do_blacklist(self, inp):
        add_to_blacklist(inp)

    def help_blacklist(self):
        print("add a domain to the blacklist.")

    def do_whitelist(self, inp):
        whitelist(inp)

    def help_whitelist(self):
        print("remove a domain from the blacklist")

    def do_clear(self, inp):
        os.system('clear')

    def help_clear(self, inp):
        print("clear the terminal")

    def default(self, inp):
        if inp == 'x' or inp == 'q':
            return self.do_exit(inp)
        else:
            print("Unknown command {}".format(inp))
            
    do_quit = do_exit
    help_quit = help_exit

    do_EOF = do_exit
    help_EOF = help_exit

def timing_lru_cache(seconds: int, maxsize: int = 128):
    def wrapper_cache(func):
        func = lru_cache(maxsize=maxsize)(func)
        func.lifetime = timedelta(seconds=seconds)
        func.expiration = datetime.utcnow() + func.lifetime

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            if datetime.utcnow() >= func.expiration:
                func.cache_clear()
                func.expiration = datetime.utcnow() + func.lifetime

            return func(*args, **kwargs)

        return wrapped_func

    return wrapper_cache

def time_and_calls_lru_cache(seconds: int, allowedCalls: int, maxsize: int = 128):
    def wrapper_cache(func):
        func = lru_cache(maxsize=maxsize)(func)
        func.lifetime = timedelta(seconds=seconds)
        func.expiration = datetime.utcnow() + func.lifetime
        func.maxIterations = allowedCalls
        func.iterations = 0

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            if datetime.utcnow() >= func.expiration:
                func.cache_clear()
                func.expiration = datetime.utcnow() + func.lifetime
            elif func.iterations >= allowedCalls:
                func.cache_clear()
                func.expiration = datetime.utcnow() + func.lifetime
                func.iterations = 0
            func.iterations += 1
            return func(*args, **kwargs)

        return wrapped_func

    return wrapper_cache

def createSocket():
    ### Creating and setting up the socket ###
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR | socket.SO_REUSEPORT, 1)
    return server_sock

def bindAndListen(server_sock, port):
    ### Binding the socket ###
    server_address = ('localhost', port)
    logging.info('starting up on %s port %s' % server_address)
    server_sock.bind(server_address)

    ### Setting the socket to listen with backlog of 5 ###
    server_sock.listen(5)

def requestLoop(client_sock):
    request = bytes()
    while True:
        data = client_sock.recv(1024)

        ### We have two variables that are receiving decoded data, we need two so that we can search on for an end of http request and one to store the entire request ###
        decodedData = data
        request += decodedData
        if b"\r\n\r\n" in decodedData:
            break
    return request

def secureRequestLoop(client_sock):
    try:
        request = bytes()
        client_sock.settimeout(0.5)
        while True:
            data = client_sock.recv(1024)
            request += data
    except socket.timeout:
        client_sock.settimeout(0)
        return request

def splitAndFormatRequest(request):
    ### Here, we are splitting the request into headers and the main request so that we can utilize the data to create a new GET request that we can use ###
    getRequest, headers = request.split(b'\r\n', 1)

    headers = headers.split(b'\r\n')

    keys = []
    values = []

    for header in headers:
        if b': ' in header:
            key, value = header.split(b': ', 1)
            keys.append(key)
            values.append(value)

    headers = dict(zip(keys, values))

    ### Creating a new GET request for the server ###
    getRequest += b"\r\n\r\n"

    ### removing the host from the GET request as it will cause an error ###
    removeHost = b"http://"
    removeHost += headers[b'Host']
    newRequest = getRequest.replace(removeHost, b'')

    return newRequest, headers

### Cache refeshes elements after 24 hours or if the size is greater than 50 ###
@timing_lru_cache(86400, maxsize=50)
def getFirstAddress(host):
    if b'443' not in host:
        ### were using getaddrinfo to get the ip address of the domain that is stored in socket ###
        ips = socket.getaddrinfo(host, 80, proto=socket.IPPROTO_TCP)

        ### we use the first address to connect to the server ###
        web_address = ips[0][4]
        return web_address
    else:
        newhost = host.replace(b':443', b'')
        ips = socket.getaddrinfo(newhost, 443, proto=socket.IPPROTO_TCP)
        
        web_address = ips[0][4]
        return web_address

### Cache refreshes elements after 15mins, if the same thing is called 4 times or if the size of the cache is greater than 128 ###
@time_and_calls_lru_cache(900, 4)
def acceptLoop(web_sock):
    try:
        toSendBack = bytes()
        web_sock.settimeout(0.5)

        ### Loop for accepting data from server ###
        while True:
            data = web_sock.recv(1024)
            toSendBack += data
            if len(data) == 0:
                break
        web_sock.settimeout(0)
        return toSendBack
    except socket.timeout:
        web_sock.settimeout(0)
        return toSendBack

def secureAcceptLoop(web_sock):
    try:
        toSendBack = bytes()
        web_sock.settimeout(0.5)

        while True:
            data = web_sock.recv(1024)
            toSendBack += data
            if len(data) == 0:
                break
        web_sock.settimeout(0)
        return toSendBack
    except socket.timeout:
        web_sock.settimeout(0)
        return toSendBack

def queryServer(web_address, getRequest):
    with createSocket() as web_sock:
        web_sock.connect(web_address)
        web_sock.sendall(getRequest)

        toSendBack = acceptLoop(web_sock)

    return toSendBack

def secureQueryServer(web_address, client_sock, getRequest, host):
    with createSocket() as web_sock:
        web_sock.connect(web_address)
        response = "HTTP/1.1 200 Connection Established\r\n\r\n"
        client_sock.send(response.encode())

        while True:
            response = secureRequestLoop(client_sock)

            web_sock.sendall(response)

            toSendBack = secureAcceptLoop(web_sock)
            client_sock.sendall(toSendBack)

def add_to_blacklist(domain):
    isCorrectFormat, domain, worldWideDomain = checkCorrectFormat(domain)
    if isCorrectFormat:
        ip = getFirstAddress(domain)
        wwIP = getFirstAddress(worldWideDomain)
        if ip[0] not in blacklist:
            blacklist.append(ip[0])
            blacklist.append(wwIP[0])
    else:
        print("Incorrect sytanx used:\nblacklist \"domain\" e.g. \"google.com\"")

def whitelist(domain):
    isCorrectFormat, domain, worldWideDomain = checkCorrectFormat(domain)
    if isCorrectFormat:
        ip = getFirstAddress(domain)
        wwIP = getFirstAddress(worldWideDomain)
        if ip[0] in blacklist:
            blacklist.remove(ip[0])
            blacklist.remove(wwIP[0])
    else:
        print("Incorrect sytanx used:\nwhitelist \"domain\" e.g. \"google.com\"")

def checkCorrectFormat(domain):
    if domain == "":
        return False
    if "www." not in domain:
        dmain = domain
        worldWideDomain = "www." + domain
    else:
        worldWideDomain = domain
        dmain = domain.replace("www.", '')
    return True, dmain, worldWideDomain

def createBlacklistHtml(domain):
    html = """HTTP/1.1 403 Forbidden\r\n\r\n
        <!doctype html>
        <html>
        <head><title>403 Forbidden</title></head>
        <body>
        <center><h1>403 Forbidden</h1></center>
        </body>
        </html>
        """.format(site = domain)
    return(html)

def takeInBlacklist(file):
    blist = file.readlines()
    for line in blist:
        blacklist.append(line.replace("\n", ''))

def writeBlacklist(file):
    for line in blacklist:
        file.write(line + "\n")

def checkRequestType(request):
    if b'CONNECT' in request:
        return 2
    else:
        return 1

def httpProtocol(client_sock, client_address, server_sock, request, requestNo):
    logging.info("Received request: id %i \n%s" % (requestNo, request))
    getRequest, headers = splitAndFormatRequest(request)
    web_address = getFirstAddress(headers[b'Host'])
    
    if web_address[0] not in blacklist:
        toSendBack = queryServer(web_address, getRequest)
        client_sock.sendall(toSendBack)

        logging.info("Complete request on id %i\n" % requestNo)
    else:
        client_sock.sendall(createBlacklistHtml(headers[b'Host']).encode())
        logging.info("Blocked request on id %i due to being blacklisted\n" % requestNo)

def httpsProtocol(client_sock, client_address, server_sock, request, requestNo):
    logging.info("Received request: id %i \n%s" % (requestNo, request))
    connectRequest, headers = splitAndFormatRequest(request)
    web_address = getFirstAddress(headers[b'Host'])

    if web_address[0] not in blacklist:
        secureQueryServer(web_address, client_sock, connectRequest, headers[b'Host'].replace(b':443', b''))
        
        logging.info("Complete request on id %i\n" % requestNo)
    else:
        client_sock.sendall(createBlacklistHtml(headers[b'Host']).encode())
        logging.info("Blocked request on id %i due to being blacklisted\n" % requestNo)

### Main Section ###
def acceptHandler(client_sock, client_address, server_sock, numberOfRequests):
    try:
        logging.info('Connection from %s %s\n' % client_address)

        request = requestLoop(client_sock)
        numberOfRequests += 1
        requestNo = numberOfRequests

        if checkRequestType(request) == 1:
            httpProtocol(client_sock, client_address, server_sock, request, requestNo)
        else:
            httpsProtocol(client_sock, client_address, server_sock, request, requestNo)
    except socket.gaierror:
        logging.info("Request failed on id %i" % requestNo)
    finally:
        client_sock.close()

def proxy_loop():
    portno = 10000
    numberOfRequests = 0

    server_sock = createSocket()
    bindAndListen(server_sock, portno)
    logging.info("Waiting for a connection\n")

    # The main loop that will run, accepting clients and then deciding what to do with them
    while True:
        client_sock, client_address = server_sock.accept()
        accept_thread = threading.Thread(target=acceptHandler, args=(client_sock, client_address, server_sock, numberOfRequests,), daemon=True)
        accept_thread.start()

def main():
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(filename='proxy.log', format=format, filemode='w', level=logging.INFO, datefmt="%H:%M:%S %d/%m/%y")

    global blacklist 
    blacklist = []
    black_file = open("blacklist.txt", 'r')
    takeInBlacklist(black_file)

    black_file = open("blacklist.txt", 'w')
    comand = console()
    comand.blacklist_file = black_file

    proxy_thread = threading.Thread(target=proxy_loop, daemon=True)
    proxy_thread.start()

    comand.cmdloop()

if __name__ == "__main__":
    main()