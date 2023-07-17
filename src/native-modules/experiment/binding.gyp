{
  "targets": [{
    "target_name": "experiment",
    "cflags_cc": [ "-fPIC -std=c++20" ],
    "cflags!": [ "-fno-exceptions" ],
    "cflags_cc!": [ "-fno-exceptions" ],
    "ldflags": [ "-Wl,-rpath,/usr/local/cuda/targets/x86_64-linux/lib" ],
    "sources": [
      "candidates.cpp",
      "cm-precompute.cpp",
      "greeting.cpp",
      "index.cpp",
      "wrap.cpp"
    ],
    "include_dirs": [
      "<!@(node -p \"require('node-addon-api').include\")",
      "/usr/local/cuda/include"
    ],
    "library_dirs": [
      '/usr/local/cuda/lib64'
    ],
    "libraries": [ '-lcudart' ], # '-lcuda'
    "dependencies": [ "filter" ],
    "defines": [ "NAPI_CPP_EXCEPTIONS" ]
  },
  {
    "target_name": "filter",
    "type": "static_library",
    "sources": [ "filter.cu" ],
    "include_dirs": [
    ],
    "rules": [{
      'extension': 'cu',           
      'inputs': [ '<(RULE_INPUT_PATH)' ],
      'rule_name': 'cuda on linux',
      'message': "compile cuda file on linux",
      'outputs': [ '<(INTERMEDIATE_DIR)/<(RULE_INPUT_ROOT).o' ],
      'process_outputs_as_sources': 1,
      'action': [
        'bash', 'nvcc.sh', '-Xcompiler', '-fPIC', '--expt-relaxed-constexpr',
        '-lineinfo',
        '-std=c++20', '-c', '<@(_inputs)', '-o', '<@(_outputs)'
      ]
    }]
  }]
}
