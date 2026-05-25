`ifndef NAC_DEFS_VH
`define NAC_DEFS_VH

// NAC container constants.
`define NAC_HEADER_BYTES      100
`define NAC_MAGIC_N           8'h4e
`define NAC_MAGIC_A           8'h41
`define NAC_MAGIC_C           8'h43
`define NAC_VERSION_V18       8'h02

// NAC ABCD system opcodes.
`define NAC_OP_INPUT          8'd2
`define NAC_OP_OUTPUT         8'd3
`define NAC_OP_CONTROL_FLOW   8'd6
`define NAC_OP_CONVERGENCE    8'd7

// Standard low NAC operation ids registered by the reference runtime.
`define NAC_STD_PASS          8'd10
`define NAC_STD_ADD           8'd11
`define NAC_STD_SUB           8'd12
`define NAC_STD_GT            8'd13
`define NAC_STD_NEG           8'd14
`define NAC_STD_WHERE         8'd15
`define NAC_STD_CLONE         8'd16
`define NAC_STD_VIEW          8'd17
`define NAC_STD_LE            8'd18
`define NAC_STD_ARANGE        8'd19
`define NAC_STD_MUL           8'd20
`define NAC_STD_DIV           8'd21
`define NAC_STD_MATMUL        8'd22
`define NAC_STD_TRANSPOSE     8'd23
`define NAC_STD_ZEROS         8'd24
`define NAC_STD_ZEROS_LIKE    8'd25
`define NAC_STD_NEW_ZEROS     8'd26
`define NAC_STD_ONES          8'd27
`define NAC_STD_ONES_LIKE     8'd28
`define NAC_STD_NEW_ONES      8'd29
`define NAC_STD_FULL          8'd30
`define NAC_STD_FULL_LIKE     8'd31

// Internal kernel classes. For A values not covered by the fixed standard NAC
// ids, the NAC loader derives this table from the standard CMAP section.
`define NAC_KCLASS_NONE       8'd0
`define NAC_KCLASS_ELEMWISE   8'd1
`define NAC_KCLASS_MATMUL     8'd2
`define NAC_KCLASS_CONV2D     8'd3
`define NAC_KCLASS_LAYERNORM  8'd4
`define NAC_KCLASS_SOFTMAX    8'd5
`define NAC_KCLASS_EMBED      8'd6
`define NAC_KCLASS_POOL       8'd7
`define NAC_KCLASS_RESHAPE    8'd8
`define NAC_KCLASS_COMPARE    8'd9

// MMAP action codes from NAC v1.8.
`define NAC_MMAP_SAVE_RESULT  8'd10
`define NAC_MMAP_FREE         8'd20
`define NAC_MMAP_FORWARD      8'd30
`define NAC_MMAP_PRELOAD      8'd40

// Programmable DSP modes. High-level kernels such as LayerNorm and Softmax are
// decomposed by the kernel controller into several of these physical modes.
`define NAC_DSP_NOP           8'd0
`define NAC_DSP_PASS          8'd1
`define NAC_DSP_ADD           8'd2
`define NAC_DSP_SUB           8'd3
`define NAC_DSP_MUL           8'd4
`define NAC_DSP_MAC           8'd5
`define NAC_DSP_NEG           8'd6
`define NAC_DSP_GT            8'd7
`define NAC_DSP_LE            8'd8
`define NAC_DSP_RELU          8'd9
`define NAC_DSP_HSIGMOID      8'd10
`define NAC_DSP_HSWISH        8'd11
`define NAC_DSP_REDUCE_SUM    8'd12
`define NAC_DSP_REDUCE_MAX    8'd13
`define NAC_DSP_SCALE         8'd14
`define NAC_DSP_RSQRT         8'd15
`define NAC_DSP_EXP           8'd16

// MEP opcodes.
`define MEP_MODEL_RUN_STATIC  8'h80
`define MEP_MODEL_TRAIN_STEP  8'h82
`define MEP_FLOW_LOOP_START   8'ha0
`define MEP_FLOW_LOOP_END     8'ha1
`define MEP_FLOW_BRANCH_IF    8'ha8
`define MEP_EXEC_RETURN       8'hfe
`define MEP_EXEC_HALT         8'hff

`endif
