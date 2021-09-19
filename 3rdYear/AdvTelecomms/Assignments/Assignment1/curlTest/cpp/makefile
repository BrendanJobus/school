default: build

build: clean
	clang++ -Wall -o curl -l curl main.cpp util.cpp

clean:
	rm -rf curl

test: build
	./curl https://freegeoip.app/json/