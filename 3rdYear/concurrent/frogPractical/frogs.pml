#define PLACES 5

int frogs[5] =  {3,1,2,4,5};

proctype moveRight(int frog) {
	printf("FROG%d (RIGHT) AT %d\n", frog, frogs[frog]);
	int at = frogs[frog];
	end:do
		:: atomic {
			(at < PLACES) &&
			(frogs[0] == at + 1) ->
				printf("FROG%d FROM %d TO %d\n", frog, at, at + 1);
				frogs[frog] = at + 1;	
				frogs[0] = at;
				printf("EMPTY %d, FROG1@%d, FROG2@%d, FROG3@%d, FROG4@%d\n", frogs[0], frogs[1], frogs[2], frogs[3], frogs[4]);
				at = at + 1;
		}
		:: atomic {
			(at < PLACES) &&
			(frogs[0] == at + 2) ->
				printf("FROG%d FROM %d TO %d\n", frog, at, at + 2);
				frogs[frog] = at + 2;
				frogs[0] = at;
				printf("EMPTY %d, FROG1@%d, FROG2@%d, FROG3@%d, FROG4@%d\n", frogs[0], frogs[1], frogs[2], frogs[3], frogs[4]);
				at = at + 2;
		}
	od
}

proctype moveLeft(int frog) {
	printf("FROG%d (LEFT) AT %d\n", frog, frogs[frog]);
	int at = frogs[frog];
	
	end:do
		:: atomic {
			(at > 1) &&
			(frogs[0] == at - 1) ->
				printf("FROG%d FROM %d TO %d\n", frog, at, at - 1);
				frogs[frog] = at - 1;
				frogs[0] = at;
				printf("EMPTY %d, FROG1@%d, FROG2@%d, FROG3@%d, FROG4@%d\n", frogs[0], frogs[1], frogs[2], frogs[3], frogs[4]);
				at = at - 1;
		}
		:: atomic {
			(at > 1) &&
			(frogs[0] == at - 2) ->
				printf("FROG%d FROM %d TO %d\n", frog, at, at - 2);
				frogs[frog] = at - 2;					
				frogs[0] = at;
				printf("EMPTY %d, FROG1@%d, FROG2@%d, FROG3@%d, FROG4@%d\n", frogs[0], frogs[1], frogs[2], frogs[3], frogs[4]);
				at = at - 2;
		}
	od
}

init {
	printf("EMPTY %d, FROG1@%d, FROG2@%d, FROG3@%d, FROG4@%d\n", frogs[0], frogs[1], frogs[2], frogs[3], frogs[4]);

	atomic {
		run moveRight(1);
		run moveLeft(3);
		run moveRight(2);
		run moveLeft(4);
	}
}