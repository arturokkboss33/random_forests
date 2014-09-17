CC = g++
CFLAGS = -Wall -g
LDFLAGS = -lm
HEADERS_DIR = headers
DSEARCH = -I ${HEADERS_DIR}


dectree_exe : dectree_bst.o dectree_class.o dectree_test.o
	${CC} ${CFLAGS} ${DSEARCH} dectree_bst.o dectree_class.o dectree_test.o -o dectree_exe

dectree_bst.o : dectree_bst.cpp ${HEADERS_DIR}/dectree_class.h ${HEADERS_DIR}/dectree_node.h
	${CC} ${CFLAGS} ${DSEARCH} -c dectree_bst.cpp

dectree_class.o : dectree_class.cpp ${HEADERS_DIR}/dectree_class.h ${HEADERS_DIR}/dectree_bst.h headers/dectree_node.h
	${CC} ${CFLAGS} ${DSEARCH} -c dectree_class.cpp

dectree_test.o : dectree_test.cpp ${HEADERS_DIR}/dectree_class.h ${HEADERS_DIR}/dectree_bst.h
	${CC} ${CFLAGS} ${DSEARCH} -c dectree_test.cpp

clean :
	rm -rf *o dectree_exe

