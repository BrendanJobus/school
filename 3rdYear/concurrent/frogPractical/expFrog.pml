#define PLACES 5

int frogs[5] =  {3,1,2,4,5};

proctype frog(int whichFrog, goingRight) {
	int at;
	if
		:: (goingRight == true) -> 
			printf("FROG%d (RIGHT) AT %d\n", whichFrog, frogs[whichFrog]);
			at = frogs[whichFrog];
		:: else -> 
			printf("FROG%d (LEFT) AT %d\n", whichFrog, frogs[whichFrog]);
			at = frogs[whichFrog];
	fi

	end:do
			:: atomic {
				(goingRight == true) &&
				(frogs[whichFrog] < PLACES) &&
				(frogs[0] == at + 1) ->
					printf("FROG%d FROM %d TO %d\n", whichFrog, at, at + 1);
					frogs[whichFrog] = at + 1;	
					frogs[0] = at;
					printf("EMPTY %d, FROG1@%d, FROG2@%d, FROG3@%d, FROG4@%d\n", frogs[0], frogs[1], frogs[2], frogs[3], frogs[4]);
					at = at + 1;
			}
			:: atomic {
				(goingRight == true) &&
				(frogs[whichFrog] < PLACES) &&
				(frogs[0] == at + 2) ->
					printf("FROG%d FROM %d TO %d\n", whichFrog, at, at + 2);
					frogs[whichFrog] = at + 2;
					frogs[0] = at;
					printf("EMPTY %d, FROG1@%d, FROG2@%d, FROG3@%d, FROG4@%d\n", frogs[0], frogs[1], frogs[2], frogs[3], frogs[4]);
					at = at + 2;
			}
			:: atomic {
				(goingRight == false) &&
				(frogs[whichFrog] < PLACES) &&
				(frogs[0] == at - 1) ->
					printf("FROG%d FROM %d TO %d\n", whichFrog, at, at - 1);
					frogs[whichFrog] = at - 1;
					frogs[0] = at;
					printf("EMPTY %d, FROG1@%d, FROG2@%d, FROG3@%d, FROG4@%d\n", frogs[0], frogs[1], frogs[2], frogs[3], frogs[4]);
					at = at - 1;
			}
			:: atomic {
				(goingRight == false) &&
				(frogs[whichFrog] < PLACES) &&
				(frogs[0] == at - 2) ->
					printf("FROG%d FROM %d TO %d\n", whichFrog, at, at - 2);
					frogs[whichFrog] = at - 2;					
					frogs[0] = at;
					printf("EMPTY %d, FROG1@%d, FROG2@%d, FROG3@%d, FROG4@%d\n", frogs[0], frogs[1], frogs[2], frogs[3], frogs[4]);
					at = at - 2;
			}
	od
}

init {
	printf("EMPTY %d, FROG1@%d, FROG2@%d, FROG3@%d, FROG4@%d\n", frogs[0], frogs[1], frogs[2], frogs[3], frogs[4]);

	atomic {
		run frog(1, true);
		run frog(2, true);
		run frog(3, false);
		run frog(4, false);
	}
}