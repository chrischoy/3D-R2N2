#CFLAGS += -g
LDFLAGS += -lm
TARGETS = off2binvox

all: ${TARGETS}

.PHONY: clean test
clean:
	@rm -f ${TARGETS}

test:
	@rm -f model.binvox
	./${TARGETS} CAD/aeroplane/01.off model.binvox
