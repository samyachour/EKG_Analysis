#! /usr/bin/env python

def hello_command(name):
    print('Hello, %s!' % name)

if __name__ == '__main__':
    import sys
    hello_command(sys.argv[1])
