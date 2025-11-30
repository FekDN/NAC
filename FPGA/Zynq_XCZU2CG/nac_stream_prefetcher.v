`include "nac_defines.vh"

module NAC_Stream_Prefetcher #(
    parameter BURST_LEN = 16, // Number of 32-bit words per AXI transaction (e.g., 16)
    parameter FIFO_DEPTH = 64 // Must be > BURST_LEN
)(
    input  wire        clk,
    input  wire        rst_n,

    // Configuration
    input  wire        enable,       // High when ALU is in S_RUN_MATH
    input  wire [31:0] start_addr,   // Combinational input from Top
    
    // Memory Master Interface (Connects to Arbiter)
    output reg  [31:0] mem_addr,
    output reg  [7:0]  mem_len,      // Burst Length (AXI encoded: N-1)
    output reg         mem_req,      // Request Strobe
    input  wire        mem_grant,    // Address Accepted
    input  wire        mem_valid,    // Data Valid Input
    input  wire [31:0] mem_rdata,    // Data Input

    // Stream Interface (Connects to ALU)
    output wire [31:0] stream_data,
    output wire        stream_valid,
    input  wire        stream_ready  // ALU consumes data
);

    // ========================================================================
    // FIFO Implementation
    // ========================================================================
    reg [31:0] fifo_mem [0:FIFO_DEPTH-1];
    reg [$clog2(FIFO_DEPTH)-1:0] wr_ptr;
    reg [$clog2(FIFO_DEPTH)-1:0] rd_ptr;
    reg [$clog2(FIFO_DEPTH):0]   count;

    wire fifo_empty = (count == 0);
    // Refill if there is space for at least one full burst + margin
    wire need_refill = (count < (FIFO_DEPTH - BURST_LEN - 2));

    assign stream_data  = fifo_mem[rd_ptr];
    assign stream_valid = !fifo_empty;

    wire push = mem_valid;
    wire pop  = stream_valid && stream_ready;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= 0; rd_ptr <= 0; count <= 0;
        end else if (!enable) begin
            // Reset FIFO pointers when stream disabled (new operation start)
            wr_ptr <= 0; rd_ptr <= 0; count <= 0;
        end else begin
            if (push) begin
                fifo_mem[wr_ptr] <= mem_rdata;
                wr_ptr <= wr_ptr + 1;
            end
            
            if (pop) begin
                rd_ptr <= rd_ptr + 1;
            end

            if (push && !pop) count <= count + 1;
            else if (!push && pop) count <= count - 1;
        end
    end

    // ========================================================================
    // Burst Request Controller with AXI 4KB Boundary Protection
    // ========================================================================
    reg [31:0] current_dram_addr;
    reg        burst_active;
    
    // Logic to calculate safe burst length
    wire [11:0] page_offset;
    wire [12:0] bytes_until_boundary;
    wire [10:0] words_until_boundary;
    reg  [7:0]  actual_transfer_size; // 1-based count of words
    reg  [7:0]  pending_burst_size;   // Store size to increment addr later

    // 1. Calculate offset within 4KB page (Lower 12 bits)
    assign page_offset = current_dram_addr[11:0];
    
    // 2. Calculate remaining bytes: 4096 (0x1000) - offset
    assign bytes_until_boundary = 13'h1000 - {1'b0, page_offset};
    
    // 3. Convert to 32-bit words (divide by 4)
    assign words_until_boundary = bytes_until_boundary[12:2];

    // 4. Determine Burst Size
    always @(*) begin
        // If remaining space is less than desired burst, chop the burst
        if (words_until_boundary < BURST_LEN) begin
            actual_transfer_size = words_until_boundary[7:0];
        end else begin
            actual_transfer_size = BURST_LEN;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mem_req <= 0;
            mem_addr <= 0;
            mem_len <= 0;
            current_dram_addr <= 0;
            burst_active <= 0;
            pending_burst_size <= 0;
        end else if (!enable) begin
            // Setup for new stream: Latch the Combinational Start Address immediately
            current_dram_addr <= start_addr;
            mem_req <= 0;
            burst_active <= 0;
            pending_burst_size <= 0;
        end else begin
            // Pulse reset
            mem_req <= 0;

            if (!burst_active) begin
                // Check if we need data and bus is idle
                if (need_refill) begin
                    mem_addr <= current_dram_addr;
                    
                    // AXI uses N-1 encoding (e.g., 16 words -> len 15)
                    mem_len  <= actual_transfer_size - 1'b1; 
                    
                    // Store the actual size (1-based) to increment address later
                    pending_burst_size <= actual_transfer_size;
                    
                    mem_req  <= 1;
                    burst_active <= 1; // Mark transaction as started
                end
            end else begin
                // Transaction in progress
                if (mem_grant) begin
                    // Address phase done.
                    // Update address for NEXT burst using the size we actually requested.
                    // Shifting left by 2 converts word count to byte count.
                    current_dram_addr <= current_dram_addr + {22'd0, pending_burst_size, 2'b00};
                    
                    burst_active <= 0; // Ready for next request logic
                end
            end
        end
    end

endmodule