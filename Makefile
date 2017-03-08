MODULES        = src src/tools src/misc
MODULE_FILES   = $(patsubst %,%/*.js,$(MODULES))

TEST_MODULES   = $(patsubst %,%/test,$(MODULES))
TEST_FILES     = $(patsubst %,%/*.js,$(TEST_MODULES))

REPORTER       = dot
MOCHA_DIR      =./node_modules/.bin/

target: all


all:


test: $(TEST_FILES)
	@NODE_ENV=test $(MOCHA_DIR)mocha $(TEST_FILES) # --reporter $(REPORTER) 


$(TEST_FILES):



