#define NUM_THREADS 5
#define success (\
	( \
    (frogs[0] == 3) && \
    (frogs[1] == 5) && \
    (frogs[2] == 4) && \
    (frogs[3] == 1) && \
    (frogs[4] == 2)   \
    ) || ( \
    (frogs[0] == 3) && \
    (frogs[1] == 5) && \
    (frogs[2] == 4) && \
    (frogs[3] == 2) && \
    (frogs[4] == 1)   \
    ) || ( \
    (frogs[0] == 3) && \
    (frogs[1] == 4) && \
    (frogs[2] == 5) && \
    (frogs[3] == 1) && \
    (frogs[4] == 2)   \
    ) || ( \
    (frogs[0] == 3) && \
    (frogs[1] == 4) && \
    (frogs[2] == 5) && \
    (frogs[3] == 2) && \
    (frogs[4] == 1)   \
    ) \
    )

ltl { []!success }

int frogs[5] = {3,1,2,4,5};
bool done = false;


mtype = { locked, unlocked };

mtype mutex = unlocked;
int mid = 0;
int init_id;

inline lock(m) { atomic{ (m==unlocked) -> m = locked; mid=_pid } }

inline unlock(m) {
	atomic{
		assert(mid==_pid);
		m = unlocked;
		mid=init_id
	}
}

inline complete() {
	if
		:: (frogs[0] == 3) && (frogs[1] == 5) && (frogs[2] == 4) && (frogs[3] == 1) && (frogs[4] == 2) -> done = true;
		:: (frogs[0] == 3) && (frogs[1] == 5) && (frogs[2] == 4) && (frogs[3] == 2) && (frogs[4] == 1) -> done = true;
		:: (frogs[0] == 3) && (frogs[1] == 4) && (frogs[2] == 5) && (frogs[3] == 1) && (frogs[4] == 2) -> done = true;
		:: (frogs[0] == 3) && (frogs[1] == 4) && (frogs[2] == 5) && (frogs[3] == 2) && (frogs[4] == 1) -> done = true;
		:: else ->
	fi
}

proctype frog(int whichFrog, goingRight) {
	int temp;

	if
		:: (goingRight == true) -> printf("FROG%d (RIGHT) AT %d\n", whichFrog, frogs[whichFrog]);
		:: else -> printf("FROG%d (LEFT) AT %d\n", whichFrog, frogs[whichFrog]);
	fi

	loop:

		lock(mutex);

		if
			:: (goingRight == true) ->
				if
					:: (frogs[0] - frogs[whichFrog] <= 2) && (frogs[0] - frogs[whichFrog] > 0) ->
						printf("FROG%d FROM %d TO %d\n", whichFrog, frogs[whichFrog], frogs[0]);
						temp = frogs[0];
						frogs[0] = frogs[whichFrog];
						frogs[whichFrog] = temp;
						printf("EMPTY %d, FROG1@%d, FROG2@%d, FROG3@%d, FROG4@%d\n", frogs[0], frogs[1], frogs[2], frogs[3], frogs[4]);

					::	else ->
				fi
			:: else ->
				if
					:: (frogs[whichFrog] - frogs[0] <= 2) && (frogs[whichFrog] - frogs[0] > 0) ->
						printf("FROG%d FROM %d TO %d\n", whichFrog, frogs[whichFrog], frogs[0]);
						temp = frogs[0];
						frogs[0] = frogs[whichFrog];
						frogs[whichFrog] = temp;
						printf("EMPTY %d, FROG1@%d, FROG2@%d, FROG3@%d, FROG4@%d\n", frogs[0], frogs[1], frogs[2], frogs[3], frogs[4]);

					:: else ->
				fi
		fi

		complete();
		if
			:: (done == true) ->
				end:
				//goto close;
			:: else ->
		fi

		unlock(mutex);

	goto loop;
	//close:
	//end:
}

init {
	init_id = _pid;
	mid = _pid;

	printf("EMPTY %d, FROG1@%d, FROG2@%d, FROG3@%d, FROG4@%d\n", frogs[0], frogs[1], frogs[2], frogs[3], frogs[4]);

	run frog(1, true);
	run frog(2, true);
	run frog(3, false);
	run frog(4, false);

	//(_nr_pr == 1);
	assert(!done);
}