# Exercise 00

A simple end-to-end test of the CSU34016 exercise system.

## Prerequisite

Haskell `stack` is installed on the machine you are using.


## Task

1. Open a command-line window and navigate to folder `Exercise00`.
2. Enter `stack test`. 
3. If this is your first time running `stack` in one of the CSU34016 Exercise folders, you may have to wait while `stack` ensures it has access to the correct versions of both the Haskell compiler and libraries. This delay should only happen once.
4. Eventually `stack` will compile, build and test the code. A lot of logging "stuff" will scroll past, ending with something like this:

```
ex00> Test suite ExerciseZero failed
Completed 2 action(s).
Test suite failure for package ex00-0.1.0.0
    ExerciseZero:  exited with: ExitFailure 1
Logs printed to console
```
5. The test fails: read the test outcome carefully. 
6. Your task is to edit `src/Ex00.hs` to:
  1. Change the first three declarations of `name`, `idno` and `username` to contain your name, student ID, and TCD username, respectively.
  2. Get the failing test to pass. 
  3. Check your changes by running `stack test` again.
7. To submit, simply upload **only** your revised `Ex00.hs` file to Blackboard. Do **not** rename the file in any way.

## Outcomes when running `stack test`

Here we show some log files showing how different conditions
that arise after running `stack test` are reported at the console.

### All Tests Pass

***This will get full marks.***


See `AllTestsPass.log`. Here the things to look out for are the occurence of `[OK]` with every tests, and the last two lines:

```
ex00> Test suite ExerciseZero passed
Completed 2 action(s).
```


### Tests Fails

***This will get less than full marks.***

See `TestsFail.log`. Here the things to look out for are the occurrence of `[Failed]` with at least one test, and the last five lines:

```
ex00> Test suite ExerciseZero failed
Completed 2 action(s).
Test suite failure for package ex00-0.1.0.0
    ExerciseZero:  exited with: ExitFailure 1
Logs printed to console
```

The line `Completed 2 action(s).` indicates that the code compiled OK.

### Fails to Compile

***This will get a mark of ZERO, 
regardless of how many lines of code were written, or days spent coding for this exercise***.

See `FailsToCompile.log`.
The key things to look for that indicates something is wrong are:

A Haskell compiler
error message, such as the following:

```
/Users/butrfeld/REPOS/CSU34016-2021/Exercise00/src/Ex00.hs:5:14: error:
    • No instance for (Num String) arising from the literal ‘01234567’
    • In the expression: 01234567
      In an equation for ‘idno’: idno = 01234567
  |
5 | idno      =  01234567   -- replace with your student id
  |              ^^^^^^^^
```

Note the occurence of the phrase `error:` at the end of the first line. 
Messages from the compiler that have the phrase `warning:` are OK, 
although you should try to address the concerns raised. 

A key indicator is in the last two lines:

```
    Process exited with code: ExitFailure 1
Progress 1/2
```

The line `Progress 1/2` indicates that first action, compilation, was performed,
but it failed, so the second action to run the tests did not proceed.


