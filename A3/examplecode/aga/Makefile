HDRS = const.h type.h extern.h exfunc.h

OBJS = super_xover.o f0.o gen.o init.o main.o  rep.o \
	stats.o  random.o  select.o utils.o halve.o scale.o


CFLAGS =  -O1
CLIBS = -lm 

ga	: $(OBJS)
	cc -o ga $(OBJS) $(CLIBS)

$(OBJS)	: $(HDRS)



