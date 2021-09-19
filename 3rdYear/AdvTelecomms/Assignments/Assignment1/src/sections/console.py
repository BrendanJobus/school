from cmd import Cmd

class MyPrompt(Cmd):
    prompt = 'pb> '
    intro = "Welcome! Type ? to list commands"

    def do_exit(self, inp):
        '''exit the application.'''
        print("Goodbye")
        return True

    def help_exit(self):
        print('exit the application. Shorthand: x q Ctrl-D.')

    def do_quit(self, inp):
        '''exit the application.'''
        print("Goodbye")
        return True

    def help_quit(self):
        print('exit the application. Shorthand: x q Ctrl-D.')

    def default(self, inp):
        if inp == 'x' or inp == 'q':
            return self.do_exit(inp)

    do_EOF = do_exit
    help_EOF = help_exit

MyPrompt().cmdloop()
print("after")