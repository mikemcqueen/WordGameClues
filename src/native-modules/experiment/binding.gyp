{
  'target_defaults': {
    'default_configuration': 'Release',
    'configurations': {
      'Common': {
        'abstract': 1,
        'cflags_cc': [ '-fPIC -std=c++23 -Wno-unused-function -Wall -Wextra -pedantic -Werror -Wconversion' ],
        'cflags!': [ '-fno-exceptions' ],
        'cflags_cc!': [ '-fno-exceptions' ],
        'include_dirs': ['../../../wtf-threadpool'], # TODO: some variable for root?
        'ldflags': [ '-Wl,-rpath,/usr/local/cuda/targets/x86_64-linux/lib' ],
      },
      'Debug': {
        'inherit_from': ['Common'],
        'defines': [ 'DEBUG', '_DEBUG' ]
      },
      'Release': {
        'inherit_from': ['Common']
      }
    }
  },
  'targets': [{
    'target_name': 'experiment',
    'type': 'none',
    'actions': [{
      'action_name': 'copy_plugin',
      'inputs': [ '<(PRODUCT_DIR)/experiment.node' ],
      'outputs': [ '<(PRODUCT_DIR)/..' ],
      'action': [ 'sh', '-c', 'cp <(_inputs) <(_outputs)' ]
    }],
    'dependencies': [ 'build_experiment' ],
  },
  {
    'target_name': 'build_experiment',
    'product_name': 'experiment',
    'sources': [
      'candidates.cpp',
      'clue-manager.cpp',
      'cm-precompute.cpp',
      'combo-maker.cpp',
      'components.cpp',
      'deferred-source.cpp',
      'cuda-common.cpp',
      'filter-stream.cpp',
      'filter-support.cpp',
      'index.cpp',
      'known-sources.cpp',
      'merge-support.cpp',
      'name-count.cpp',
      'source-core.cpp',
      'unique-variations.cpp',
      'unwrap.cpp',
      'validator.cpp',
      'wrap.cpp',
    ],
    'include_dirs': [
      '<!@(node -p \'require("node-addon-api").include\')',
      '/usr/local/cuda/include'
    ],
    'library_dirs': [
      '/usr/local/cuda/lib64'
    ],
    'libraries': [
      '-Wl,--whole-archive',
      '<(PRODUCT_DIR)/kernels_lib.a',
      '-Wl,--no-whole-archive',
      '<(PRODUCT_DIR)/kernels_dlink.a',
      '-lcudart',
      '-lcudadevrt'
    ],
    'dependencies': [
      'dlink_kernels'
    ],
    'defines': [ 'NAPI_CPP_EXCEPTIONS' ]
  },
  {
    'target_name': 'dlink_kernels',
    'type': 'none',
    'sources': [
      '<(PRODUCT_DIR)/kernels_lib.a',
    ],
    'rules': [{
      'extension': 'a',
      'rule_name': 'device link kernels',
      'message': 'device link kernels on linux',
      'variables': {
        'obj_files': [
          'merge.o',
          'filter.o',
          'or-filter.o',
        ]
      },
      'inputs': [ '<(RULE_INPUT_PATH)' ],
      'outputs': [
        '<(SHARED_INTERMEDIATE_DIR)/kernels_dlink.a'
      ],
      'action': [
        'env', 'OBJ_DIR=<(SHARED_INTERMEDIATE_DIR)', 'LIB_DIR=<(PRODUCT_DIR)',
        'make', '-sf', 'kernel.mk', '<(SHARED_INTERMEDIATE_DIR)/kernels_dlink.a'
      ]
    }],
    'dependencies': [ 'kernels_lib' ],
  },
  {
    'target_name': 'kernels_lib',
    'type': 'none',
    'actions': [{
      'action_name': 'create_kernels_lib',
      'inputs': [
        '<(SHARED_INTERMEDIATE_DIR)/merge.o',
        '<(SHARED_INTERMEDIATE_DIR)/or-filter.o',
        '<(SHARED_INTERMEDIATE_DIR)/filter.o',
      ],
      'outputs': [ '<(PRODUCT_DIR)/kernels_lib.a' ],
      'action': [
        'ar', 'rcs', '<@(_outputs)', '<@(_inputs)'
      ]
    }],
  },
  {
    'target_name': 'old_kernels_lib',
    'type': 'static_library',
    'sources': [
      '<(SHARED_INTERMEDIATE_DIR)/merge.o',
      '<(SHARED_INTERMEDIATE_DIR)/or-filter.o',
      '<(SHARED_INTERMEDIATE_DIR)/filter.o',
    ],
  },
  {
    'target_name': 'unused_compile_kernels',
    'type': 'none',
    'sources': [
      'merge.cu',
      'or-filter.cu',
      'filter.cu',
    ],
    'rules': [{
      'extension': 'cu',           
      'rule_name': 'compile kernels',
      'message': 'compile CUDA kernels on linux',
      'inputs': [
        '<(RULE_INPUT_PATH)',
        '<(SHARED_INTERMEDIATE_DIR)/<(RULE_INPUT_ROOT).d'
      ],
      'outputs': [ '<(SHARED_INTERMEDIATE_DIR)/<(RULE_INPUT_ROOT).o' ],
      'action': [
          'env', 'OBJ_DIR=<(SHARED_INTERMEDIATE_DIR)', 'FILE=<(RULE_INPUT_ROOT)',
          'make', '-sf', 'kernel.mk', 'compile'
      ]
    }]
  }]
}
