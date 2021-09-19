import logging, os, socket, string, threading, time
from cmd import Cmd
from collections import OrderedDict
from datetime import datetime, timedelta
from functools import lru_cache, wraps

### LRU Cache work ###
class LRUCache:
    # initialises the cache with a max seconds allowed in the cache, max number of calls allowed before a key cannot be called from the cache, and a max size of the cache
    def __init__(self, seconds: int, callsAllowed: int, capacity: int = 128):
        self.cache = OrderedDict()
        self.timeAllowed = timedelta(seconds = seconds)
        self.callsAllowed = callsAllowed
        self.capacity = capacity

    # check if the key is in the cache, has not been called too many times, and has not been in the cache over the allowed duration, if all these things are ok
    # then we pass back the data, otherwise we return -1
    def get(self, key: bytes) -> bytes:
        if key not in self.cache:
            return -1
        else:
            if self.cache.get(key)[1] < self.callsAllowed:
                if datetime.utcnow() < self.cache.get(key)[2]: 
                    self.cache.move_to_end(key)
                    self.cache.get(key)[1] += 1
                    return self.cache[key][0]
        return -1

    # we put the key and value into the cache, we add in a counter and end time for how many times this item has been called, and when it will expire in the cache
    # we add this new value with its key to the end of the cache, if the cache is larger than the capacity allowed, we pop the first item off
    def put(self, key: bytes, value: bytes) -> None:
        valueToAdd = [value, 0, datetime.utcnow() + self.timeAllowed]
        self.cache[key] = valueToAdd
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last = False)

### Caching wrapper ###

# a wrapper for the built in python lru cache that adds an expiration date to a cached item
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

### The Proxy ###
# The proxy class has been structured such that, the first section of functions can be viewed
# to get a good idea of how the proxy server was implemented, and the second section holds
# the functions that do the actual detailed work, e.g. creating and binding sockets.
class Proxy:
    # gives itself a cache for web pages as well as sets a counter for the number of requests it received so that it can id requests
    def __init__(self, port):
        self.portno = port
        self.numberOfRequests = 1
        self.web_cache = LRUCache(900, 3)

    ### Structure ###
    
    # This is the main proxy function that will listen into the port specified, once a connection comes in from a client, it calls a thread to handle that connection
    def start(self):
        server_sock = self.createSocket()
        self.bindAndListen(server_sock, self.portno)
        logging.info("Waiting for a connection\n")

        # The main loop that will run, accepting clients and then deciding what to do with them
        while True:
            client_sock, client_address = server_sock.accept()
            accept_thread = threading.Thread(target=self.acceptHandler, args=(client_sock, client_address,), daemon=True)
            accept_thread.start()

    # This handler function will take in the request from the client and then pass it off to the appropriate protocol
    def acceptHandler(self, client_sock, client_address):
        try:
            request = self.intakeData(client_sock)
            requestNo = self.numberOfRequests
            self.numberOfRequests += 1

            # this is checking to make sure that a non empty request has been sent, if it is empty, just discard it
            if request != -1:
                reqType = self.checkRequestType(request)

                # this checks the protocol as well as wether we are going to need to send data with the request (e.g. POST), as well as if it is well formed
                # if it is not well formed, just discard it
                if reqType == 0:
                    self.httpProtocol(client_sock, client_address, request, requestNo, reqType)
                elif reqType == 1:
                    self.httpProtocol(client_sock, client_address, request, requestNo, reqType)
                elif reqType == 2:
                    self.httpsProtocol(client_sock, client_address, request, requestNo)
        # this handles address related errors that might come up in the protocols
        except socket.gaierror:
            logging.info("Request failed on id %i\n" % requestNo)
        finally:
            client_sock.close()

    ### HTTP functions ###
    # This is the main http protocol it will get the host name from the request and then send it off with the request to the send request function
    # it will receive back the data we wend to the client as well as the bandwidth used, we start a timer here for how long the connection lasts
    def httpProtocol(self, client_sock, client_address, request, requestNo, reqType):
        logging.info(f"Connecting from {client_address}\nRequest: id {requestNo} \n{request}\n")
        startTime = time.time()
        request, headers = self.getHost(request)

        toSendBack, bandwidth = self.sendRequest(request, headers[b'Host'])
        
        if toSendBack != -1:
            client_sock.sendall(toSendBack)
            bandwidth += len(toSendBack)
            logging.info(f"Completed request on id {requestNo}, time taken {time.time() - startTime}, bandwidth used {bandwidth} bytes\n")
        else:
            forbiddenHTML = self.createBlacklistHtml(headers[b'Host']).encode()
            client_sock.sendall(forbiddenHTML)
            logging.info(f"Blocked request on id {requestNo} due to being blacklisted, time taken {time.time() - startTime}, bandwidth used {len(forbiddenHTML)} bytes\n")

    # here, we are going to get the address, check if it is blacklisted, it its not, then we check if we can get the page from the cache, if we can't, we query the web server
    # and put page that we get back from the server onto the cache, and send back the page and the bandwidth used
    def sendRequest(self, request, host):
        web_address = self.getFirstAddress(host)
        if web_address[0] not in blacklist:
            page = self.web_cache.get(request)
            if page != -1:
                return page, 0
            else:
                toSendBack, bytesUsed = self.queryServer(web_address, request)
                self.web_cache.put(request, toSendBack)
                return toSendBack, bytesUsed
        else:
            return -1, 0

    # this is the function that sends the request, bandwidth used is the size of the request and the size of the data that received back
    def queryServer(self, web_address, request) -> (bytes, int):
        with self.createSocket() as web_sock:
            web_sock.connect(web_address)
            web_sock.sendall(request)

            toSendBack = self.intakeData(web_sock)

        return toSendBack, len(request) + len(toSendBack)

    ### HTTPS functions

    # This is the main https protocol, we are still going to time this just for debugging purposes
    # we get the web address from the host, and then if it is not blacklisted, we create a connection to the web socket, send back an ok to the client, and start tunneling
    def httpsProtocol(self, client_sock, client_address, request, requestNo):
        logging.info(f"Connecting from {client_address}\nRequest: id {requestNo} \n{request}\n")
        startTime = time.time()
        _, headers = self.getHost(request)
        web_address = self.getFirstAddress(headers[b'Host'])

        if web_address[0] not in blacklist:
            with self.createSocket() as web_sock:
                web_sock.connect(web_address)
                response = "HTTP/1.1 200 Connection Established\r\n\r\n"
                client_sock.send(response.encode())
                self.tunnelData(web_sock, client_sock, headers[b'Host'].replace(b':443', b''))
            
            logging.info(f"Complete request on id {requestNo}, time taken {time.time() - startTime} seconds\n")
        else:
            client_sock.sendall(self.createBlacklistHtml(headers[b'Host']).encode())
            logging.info(f"Blocked request on id {requestNo} due to being blacklisted\n")

    # Here we simply send data back and forth between the client and web server, if we receive an empty byte, then we close the tunnel
    def tunnelData(self, web_sock, client_sock, host):
        while True:
            client_data = self.intakeData(client_sock)
            if client_data == -1:
                break
            web_sock.sendall(client_data)

            web_data = self.intakeData(web_sock)
            if web_data == -1:
                break
            client_sock.sendall(web_data)

    ### Implementing ###
    
    # Gets the first address we can find of a host, caches it for one day so long as the cache doesn't grow to a size greater than 50
    @timing_lru_cache(86400, maxsize=50)
    def getFirstAddress(self, host):
        try:
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
        except TypeError:
            if '443' not in host:
                ### were using getaddrinfo to get the ip address of the domain that is stored in socket ###
                ips = socket.getaddrinfo(host, 80, proto=socket.IPPROTO_TCP)

                ### we use the first address to connect to the server ###
                web_address = ips[0][4]
                return web_address
            else:
                newhost = host.replace(':443', '')
                ips = socket.getaddrinfo(newhost, 443, proto=socket.IPPROTO_TCP)
                
                web_address = ips[0][4]
                return web_address

    # Checks if the request is a GET, POST or CONNECT request, if its not, return -1
    def checkRequestType(self, request):
        if b'CONNECT' in request:
            return 2
        elif b'POST' in request:
            return 1
        elif b'GET' in request:
            return 0
        else:
            return -1

    # Creates a socket that is reusable
    def createSocket(self):
        ### Creating and setting up the socket ###
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR | socket.SO_REUSEPORT, 1)
        return sock

    # binds a socket to port and allows up to 20 connections to be waiting to be accepted
    def bindAndListen(self, server_sock, port):
        ### Binding the socket ###
        server_address = ('localhost', port)
        logging.info('starting up on %s port %s' % server_address)
        server_sock.bind(server_address)

        ### Setting the socket to listen with backlog of 5 ###
        server_sock.listen(20)

    # This function gets the host name from a request for use in getting the ip address
    def getHost(self, request):
        ### Here, we are splitting the request into headers and the main request so that we can utilize the data to create a new GET request that we can use ###
        _, headers = request.split(b'\r\n', 1)

        headers = headers.split(b'\r\n')

        keys = []
        values = []

        for header in headers:
            if b': ' in header:
                key, value = header.split(b': ', 1)
                keys.append(key)
                values.append(value)
            elif len(header) != 0:
                ### This will be used to get to the data from POST request
                keys.append(b'Data')
                values.append(header)

        headers = dict(zip(keys, values))

        if b'Proxy-Connection: ' in request:
            rep = b'Proxy-Connection: '
            rep += headers.get(b'Proxy-Connection')
            request = request.replace(rep + b'\r\n', b'')

        return request, headers

    # This creates the forbidden HTML message used when a client attempts to access a blacklisted website
    def createBlacklistHtml(self, domain):
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

    # This function takes in data from a socket, it sets a timeout on the socket, and then receives data until it receives a timeout exception or if the length of the data received is 0, it returns -1
    def intakeData(self, sock):
        try:
            received = bytes()
            sock.settimeout(0.1)

            ### Loop for accepting data from server ###
            while True:
                data = sock.recv(10240)
                received += data
                if len(data) == 0:
                    return -1

        except socket.timeout:
            sock.settimeout(0)
            return received

### The console ###
# This was built with pythons built in Cmd, a do_KEYWORD function will be activated when the keyword is typed into the console, 
# whatever it was followed by will be passed as the optional inp parameter, a help_KEYWORD will give a description of the function when help KEYWORD is typed
# also holds the functions that deal with blacklisting towards the bottom
class console(Cmd):
    prompt = 'Proxy> '
    intro = "The Proxy has been started! Type ? for help"
    blacklist_file = ""
    clear = 'clear'
    p = Proxy(42069)

    def do_exit(self, inp):
        console.writeBlacklist(self, self.blacklist_file)
        print("Goodbye")
        return True

    def help_exit(self):
        print("exit the application. Shorthand: x q Ctrl-D.")

    def do_blacklist(self, inp):
        console.add_to_blacklist(self, inp)

    def help_blacklist(self):
        print("add a domain to the blacklist.")

    def do_whitelist(self, inp):
        console.whitelist(self, inp)

    def help_whitelist(self):
        print("remove a domain from the blacklist")

    def do_clear(self, inp):
        os.system(self.clear)

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

    # This will write the blacklist to a file when the proxy is closed so that the blacklist will persist
    def writeBlacklist(self, file):
        for line in blacklist:
            file.write(line + "\n")

    # Checks to see if the format is correct (i.e. is not empty), and then returns a domain with and without www. prefix, as these will both give different ips that
    # may be called, once it gets the ips for these domains, it appends them onto the blacklist if they are not already in the blacklist
    def add_to_blacklist(self, domain):
        try:
            isCorrectFormat, domain, worldWideDomain = self.checkCorrectFormat(domain)
            if isCorrectFormat:
                ip = self.p.getFirstAddress(domain)
                wwIP = self.p.getFirstAddress(worldWideDomain)
                if ip[0] not in blacklist:
                    blacklist.append(ip[0])
                    blacklist.append(wwIP[0])
            else:
                print("Incorrect sytanx used:\nblacklist \"domain\" e.g. \"google.com\"")
        except socket.gaierror:
            print("No address associated with the hostname")

    # Does the exact same thing that add_to_blacklist does, but removes them from the blacklist if they are in the it
    def whitelist(self, domain):
        try:
            isCorrectFormat, domain, worldWideDomain = self.checkCorrectFormat(domain)
            if isCorrectFormat:
                ip = self.p.getFirstAddress(domain)
                wwIP = self.p.getFirstAddress(worldWideDomain)
                if ip[0] in blacklist:
                    blacklist.remove(ip[0])
                    blacklist.remove(wwIP[0])
            else:
                print("Incorrect sytanx used:\nwhitelist \"domain\" e.g. \"google.com\"")
        except socket.gaierror:
            print("No address associated with the hostname")

    # checks the format, if it has a www. prefix, creates another domain without the prefix, if it does not have the prefix, creates another domain with the prefix
    def checkCorrectFormat(self, domain):
        if domain == "":
            return False
        if "www." not in domain:
            dmain = domain
            worldWideDomain = "www." + domain
        else:
            worldWideDomain = domain
            dmain = domain.replace("www.", '')
        return True, dmain, worldWideDomain

# Function that takes in the blacklisting file and places all of the ips in it into a list that the proxy uses
def takeInBlacklist(file):
    blist = file.readlines()
    for line in blist:
        blacklist.append(line.replace("\n", ''))

# The mainline, this is called when the program is first run, starts the proxy thread and the console loop
def main():
    # Setting the logging format, should be the date and time followed by our message, the date should be hour:minute:second day/month/year
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(filename='proxy.log', format=format, filemode='w', level=logging.INFO, datefmt="%H:%M:%S %d/%m/%y")

    # creating a global blacklist that both the proxy and console are going to use
    global blacklist
    blacklist = []
    # Try to open blacklist.txt to read in blacklisted ips, if the file does not exits, just move on
    try:
        black_file = open("blacklist.txt", 'r')
        takeInBlacklist(black_file)
    except FileNotFoundError:
        pass
    black_file = open("blacklist.txt", 'w')

    command = console()
    command.blacklist_file = black_file

    # This program was written for linux, this checks if we're on windows, and if we are, fixes some errors
    if os.name == 'nt':
        socket.SO_REUSEPORT = socket.SO_REUSEADDR
        command.clear = 'cls'

    # Create a proxy instance on port 10000, run it on proxy_thread, daemon=True will run the thread till the main thread ends
    proxy = Proxy(10000)
    proxy_thread = threading.Thread(target=proxy.start, daemon=True)
    proxy_thread.start()

    command.cmdloop()

if __name__ == "__main__":
    main()