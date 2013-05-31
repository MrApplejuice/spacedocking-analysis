SRC_FILES=$(shell find -name '*.cpp')
ADDITIONAL_CODE_FILES=$(shell find -name '*.h*') $(shell find -name '*.inl')

FLAGS=-pthread -I. -lboost_program_options -lboost_thread -lboost_filesystem

cluster: $(SRC_FILES) $(ADDITIONAL_CODE_FILES)
	g++ $(FLAGS) $(SRC_FILES) -o $@
