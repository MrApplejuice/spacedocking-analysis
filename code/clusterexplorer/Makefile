.PHONY: all allall release linux clean deps invalid-target no-target
.SUFFIXES:

# Forward rule for "all"
all: allall

# Directories to search for every system
ifeq (,$(GAME))
$(error GAME must be specified - the game to compile)
endif

include $(GAME).makefile

# Sanity checks if game defines are correct
ifeq (,$(GAME_NAME))
$(error GAME_NAME must be specified)
endif

ifeq (,$(GAME_SRC) $(GAME_SRC_DIR))
$(error GAME_SRC or GAME_SRC_DIR must be specified)
endif

ifeq (,$(GAME_GL_VERSION))
$(error GAME_GL_VERSION must be specified)
endif

# Source directories for the game
GENERAL_SRC_DIRECTORIES=engine/ $(GAME_SRC_DIR)

# Src file extensions to search for and compile
SRC_EXTENSIONS=*.cpp *.c

# Default directory for game resources
BUILDDIR=bin/$(GAME_NAME)/
DEPSDIR=deps/$(GAME_NAME)/

GCC=gcc
CC=cc
CPP=g++

OPENGL_DEF=-DOPENGL_VERSION=$(GAME_GL_VERSION)
VERSION_DEF=$(addprefix -D, $(shell buildhelp/parseVersionDefinition.sh $(GAME_VERSION)))

ALLTARGETS=
PREREQUISITES=$(GAME_EXTRA_TARGETS)
TARGETPREREQUISITES=
TARGETPOSTREQUISITES=

CLEAN_EXTRAS=$(GAME_EXTRA_CLEAN)

# Accepted target systems
ACCEPTED_SYSTEMS=$(shell ls buildhelp/systems/)

# Determine the target for rule "all"
ALLTARGETS=
# Determine which "all-targets" do not need make to include dependencies
NODEPSTARGETS=clean invalid-target no-target

# Converted goals include all makecmdgoals but translate an empty makecmdgoal to "notarget"
CONVERTEDGOALS:=$(MAKECMDGOALS)
ifeq (,$(filter $(ACCEPTED_SYSTEMS), $(CONVERTEDGOALS)))
CONVERTEDGOALS:=notarget
endif

# Use uname to determine system if and only if notarget has been specified 
ifneq (,$(filter notarget,$(CONVERTEDGOALS)))
SYSTEM=$(shell uname | tr '[A-Z]' '[a-z]' | sed 's/mingw.._.*/mingw-win/')
else
SYSTEM=$(CONVERTEDGOALS)
endif

# Add release configurations
ifneq (,$(filter $(MAKECMDGOALS),release))
FLAGS+=-DNDEBUG -O3
RELEASE=yes
else
RELEASE=
endif

SRC_DIRECTORIES=$(GENERAL_SRC_DIRECTORIES)


# Test if a valid system has been specified, Include system spcific settings if it has been
ifeq (,$(SYSTEM))
BUILDTARGET=invalid-target
CONVERTEDGOALS+=invalid-target
ALLTARGETS=
else
include buildhelp/systems/$(SYSTEM)
endif

# Test if clean has been specified
ifneq (,$(filter clean,$(MAKECMDGOALS)))
CONVERTEDGOALS+=invalid-target
TARGETPREREQUISITES=
ALLTARGETS=
else
TARGETPREREQUISITES+=$(PREREQUISITES)
endif

# Collect all source files
SRC_FILES=$(subst //,/,$(foreach e,$(SRC_EXTENSIONS),$(foreach d,$(SRC_DIRECTORIES),$(shell find $(d) -name '$(e)'))))
SRC_FILES+=$(GAME_SRC)

# Determine dep-files and objfiles belonging to collected source files
DEPS_FILES=$(addprefix $(DEPSDIR),$(addsuffix .d,$(SRC_FILES)))
OBJ_FILES=$(addprefix $(BUILDDIR),$(addsuffix .o,$(basename $(SRC_FILES))))

FLAGS+=$(GAME_FLAGS) -DTARGET_PLATFORM_NAME="\"$(TARGET_PLATFORM_NAME)\"" $(OPENGL_DEF) $(VERSION_DEF)
CINCLUDES+=. includes/
INCLUDE_FLAGS=$(addprefix -I,$(CINCLUDES))
CFLAGS+=$(FLAGS) -DGL_GLEXT_PROTOTYPES $(INCLUDE_FLAGS)
CPPFLAGS+=$(FLAGS) -DGL_GLEXT_PROTOTYPES $(INCLUDE_FLAGS)
LINKFLAGS+=$(addprefix -L,$(LIBPATHS)) $(addprefix -l,$(LIBS))
LINKPREFLAGS+=
LINKER?=$(GCC)

allall: all-msg $(ALLTARGETS)

all-msg:
	@echo "No target specified - uname identified: "$(SYSTEM)

release:
	@echo "Set release configuration"

clean:
	-rm $(BUILDTARGET)
	-rm -r $(BUILDDIR)
	-rm -r $(DEPSDIR)
ifneq (,$(CLEAN_EXTRAS))
	-rm -r $(CLEAN_EXTRAS)
endif
ifneq (,$(CLEAN_MAKE_DIRS))
	-$(foreach d,$(CLEAN_MAKE_DIRS),$(shell cd $d && make clean))
endif

# Error target
invalid-target:
	@echo "Invalid build target; valid targets are: "$(ACCEPTED_SYSTEMS)

# General rules
deps: $(DEPS_FILES)

# Make build paths
$(BUILDDIR): 
	@-mkdir -p $(BUILDDIR)

$(DEPSDIR): 
	@-mkdir -p $(DEPSDIR)

# Check if we have a valid build target. If we have one, extract path and create path
ifeq (,$(filter $(BUILDTARGET),invalid-target no-target))
ifneq (,$(dir $(BUILDTARGET)))
$(dir $(BUILDTARGET)):
	mkdir -p $@
endif

#  Deps
$(DEPSDIR)%.cpp.d: %.cpp $(DEPSDIR) $(PREREQUISITES)
	@mkdir -p $(dir $@)
	$(CPP) $(CPPFLAGS) -M -MT "$(BUILDDIR)$(addsuffix .o,$(basename $<)) $(addsuffix .d,$<)" -o $@ $<

$(DEPSDIR)%.c.d: %.c $(DEPSDIR) $(PREREQUISITES)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -M -MT "$(BUILDDIR)$(addsuffix .o,$(basename $<)) $(addsuffix .d,$<)" -o $@ $<

ifneq (,$(filter $(CONVERTEDGOALS),$(NODEPSTARGETS)))
else
include $(DEPS_FILES)
endif
	
$(BUILDDIR)%.o: %.cpp $(BUILDDIR) $(DEPSDIR)/%.cpp.d $(PREREQUISITES) 
	@-mkdir -p $(dir $@)
	$(CPP) $(CPPFLAGS) -c -o $@ $<

$(BUILDDIR)%.o: %.c $(BUILDDIR) $(DEPSDIR)/%.c.d $(PREREQUISITES) 
	@-mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c -o $@ $<

# Link product
$(BUILDTARGET): $(BUILDDIR) $(OBJ_FILES) $(TARGETPREREQUISITES) $(TARGETPOSTREQUISITES)
	$(LINKER) $(LINKPREFLAGS) $(OBJ_FILES) -o $(BUILDTARGET) $(LINKFLAGS) 
endif


