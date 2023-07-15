{
  "targets": [{
    "target_name": "experiment",
    "cflags_cc": [ "-fPIC" ],
    "cflags!": [ "-fno-exceptions" ],
    "cflags_cc!": [ "-fno-exceptions" ],
    "ldflags": [ "-Wl,-rpath,/usr/local/cuda-12.2/targets/x86_64-linux/lib" ],
    "sources": [
      "candidates.cpp",
      "cm-precompute.cpp",
      "greeting.cpp",
      "index.cpp",
      "wrap.cpp"
    ],
    "include_dirs": [
      "<!@(node -p \"require('node-addon-api').include\")"
    ],
    "library_dirs": [
      '/usr/local/cuda-12.2/lib64'
    ],
#    "libraries": [ '-lcuda', '-lcudart' ],
    "libraries": [ '-lcudart' ],
    "dependencies": [ "filter" ],
    "defines": [ "NAPI_CPP_EXCEPTIONS" ]
  },
  {
    "target_name": "filter",
    "type": "static_library",
    "sources": [ "filter.cu" ],
    "include_dirs": [
#      '<@(raw_includes)'
    ],
    "rules": [{
      'extension': 'cu',           
      'inputs': [ '<(RULE_INPUT_PATH)' ],
      'rule_name': 'cuda on linux',
      'message': "compile cuda file on linux",
      'outputs': [ '<(INTERMEDIATE_DIR)/<(RULE_INPUT_ROOT).o' ],
      'process_outputs_as_sources': 1,
      'action': [
        'bash', 'nvcc.sh', '-Xcompiler', '-fPIC', '-c',
        '<@(_inputs)', '-o', '<@(_outputs)'
      ]
    }]
  }]
}
