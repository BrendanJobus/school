add r0, #4, r20

max:
	add r0, r26, r1				; v = a
	sub r27, r1, r0 {C}			; b > a
	jle	nex
	xor r0, r0, r0				; nop for pipeline efficiency
	add r0, r27, r1				; v = b
nex:
	sub r28, r1, r0 {C}			; c > v
	jle	rtrn
	xor r0, r0, r0
	add r0, r28, r1				; v = c
rtrn:
	RET r15, 0
	xor r0, r0, r0
	
max5:
	add r0, r26, r11			; i -> r11
	add r0, r27, r12			; j -> r12
	add r0, r20, r10			; inp_int -> r10
	CALLR r15, max
	
	add r0, r1, r10				; max(inp_int, i, j) -> r10
	add r0, r28, r11			; k -> r11
	add r0, r29, r12			; l -> r12
	CALLR r15, max
	
	RET r15, 0
	xor r0, r0, r0
	
fun:
	sub r27, #0, r0 {C}			; b == 0
	jne ovr
	xor r0, r0, r0
	add r0, #0, r1				; ret 0
	RET r15, 0
	xor r0, r0, r0
	
ovr:
	add r0, r27, r10			; b -> r10
	add r0, #2, r11				; #2 -> r11
	CALLR r15, div
	
	add r0, r1, r11				; b / 2 -> r11
	add r26, r26, r10			; a + a -> r10
	CALLR r15, fun				; fun(a + a, b / 2)
	
	add r0, r27, r10			; mod(b, 2) b -> r10
	add r0, #2, r11				;			#2 -> r11
	CALLR r15, mod				; calling mod(b, 2)
	sub r1, #0, r0 {C}			; b % 2 == 0
	jne nding
	xor r0, r0, r0
	
nding:
	add r1, r26, r1				; fun(a + a, b / 2) + a
	RET r15, 0					; return fun(a + a, b / 2) + a
	xor r0, r0, r0
	