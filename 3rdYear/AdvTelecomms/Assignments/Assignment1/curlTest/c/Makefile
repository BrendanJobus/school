default: build

build:clean
	clang -Wall -o curl -l curl main.c util.c

clean:
	rm -rf curl

test: build
	./curl https://freegeoip.app/json/