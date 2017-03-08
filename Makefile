MODULES        = src src/tools src/misc
MODULE_FILES   = $(patsubst %,%/*.js,$(MODULES))
MODULE_DEPEND  = $(MODULE_FILES)

TEST_MODULES   = $(patsubst %,%/test,$(MODULES))
TEST_PREFIX    = test-
TEST_FILES     = $(patsubst %,%/$(TEST_PREFIX)*.js,$(TEST_MODULES))
TEST_DEPEND    = $(patsubst %,%/*.js,$(TEST_MODULES))

REPORTER       = dot
MOCHA_DIR      =./node_modules/.bin/

target: all


all:


test: $(MODULE_DEPEND) $(TEST_DEPEND)
	@NODE_ENV=test $(MOCHA_DIR)mocha $(TEST_FILES) # --reporter $(REPORTER) 


$(MODULE_DEPEND):


$(TEST_DEPEND):




