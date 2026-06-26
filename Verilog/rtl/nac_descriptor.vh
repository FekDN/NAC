`ifndef NAC_DESCRIPTOR_VH
`define NAC_DESCRIPTOR_VH

// Descriptor layout extension. Lower 64 bits remain available for the legacy
// pointer/shape handle used by earlier tests and integration shells.
`define NAC_DESC_LEGACY_WIDTH      64
`define NAC_DESC_MIN_WIDTH         96
`define NAC_DESC_FLAG_IS_BFP       64
`define NAC_DESC_FLAG_IS_SPARSE_2_4 65
`define NAC_DESC_FLAG_IS_RLE       66
`define NAC_DESC_FLAG_IS_PALETTE   67
`define NAC_DESC_FLAG_STATIC       68

`define NAC_DESC_FLAG(DESC, BIT) DESC[BIT]

`endif
