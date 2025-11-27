`include "nac_defines.vh"

module NAC_Byte_Fetcher #(
    // 2^2 = 4 words (16 bytes). 
    // Small enough for FPGA LUTRAM/Distributed RAM, large enough to hide DDR latency.
    parameter FIFO_DEPTH_LOG2 = 2 
)(
    input  wire        clk,
    input  wire        rst_n,

    // ========================================================================
    // Control Interface (From Main FSM)
    // ========================================================================
    // Active High: invalidates current buffer, resets address to start_addr.
    // Used on Pattern CALL, RETURN, and JUMP operations.
    input  wire        flush,        
    
    // The exact byte address to jump to (can be unaligned, e.g., 0x1003)
    input  wire [31:0] start_addr,   
    
    // High if the internal buffer is empty (Decoder must stall)
    output wire        busy,         

    // ========================================================================
    // Consumer Interface (To NAC Decoder)
    // ========================================================================
    input  wire        byte_req,     // Decoder requests 1 byte
    output reg  [7:0]  byte_data,    // The extracted byte
    output reg         byte_valid,   // Data valid pulse (1 cycle)

    // ========================================================================
    // Memory Provider Interface (To Memory Arbiter)
    // ========================================================================
    output reg  [31:0] mem_addr,     // Aligned 32-bit address
    output reg         mem_req,      // Read Strobe
    input  wire        mem_grant,    // Arbiter accepted the request
    input  wire        mem_valid,    // Data returning from DDR
    input  wire [31:0] mem_rdata     // 32-bit Data Payload
);

    // Constants
    localparam FIFO_DEPTH = (1 << FIFO_DEPTH_LOG2);

    // ========================================================================
    // Internal State Registers
    // ========================================================================
    
    // 1. Address Management
    reg [31:0] next_fetch_addr; // Always aligned to 32-bit boundaries

    // 2. FIFO Storage (Distributed RAM inferred)
    reg [31:0] fifo_data [0:FIFO_DEPTH-1];
    reg [FIFO_DEPTH_LOG2-1:0] wr_ptr;
    reg [FIFO_DEPTH_LOG2-1:0] rd_ptr;
    reg [FIFO_DEPTH_LOG2:0]   words_in_fifo; // Counter: How many valid words do we have?

    // 3. Byte Alignment
    // Points to the current byte [0..3] within the word at fifo_data[rd_ptr]
    reg [1:0]  byte_offset; 

    // 4. Flow Control / In-Flight Tracking
    // Counts requests sent to DDR that haven't returned 'valid' yet.
    // Prevents FIFO overflow and helps filter stale data after a flush.
    reg [FIFO_DEPTH_LOG2:0] pending_req_cnt; 
    
    // Logic: We are busy (starved) if we have no words in FIFO.
    // Note: We might have a word, but if byte_offset wrapped, we might need next word.
    // Ideally, busy is strictly words_in_fifo == 0.
    assign busy = (words_in_fifo == 0);

    // ========================================================================
    // Main Logic
    // ========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset all state
            mem_req <= 0;
            mem_addr <= 0;
            byte_valid <= 0;
            byte_data <= 0;
            
            wr_ptr <= 0;
            rd_ptr <= 0;
            words_in_fifo <= 0;
            byte_offset <= 0;
            
            next_fetch_addr <= 0;
            pending_req_cnt <= 0;
        end 
        else if (flush) begin
            // ----------------------------------------------------------------
            // FLUSH EVENT: Handling Context Switches
            // ----------------------------------------------------------------
            // 1. Stop requesting
            mem_req <= 0;
            byte_valid <= 0;
            
            // 2. Clear Buffer Pointers
            wr_ptr <= 0;
            rd_ptr <= 0;
            words_in_fifo <= 0;
            
            // 3. Reset Pending Count
            // IMPORTANT: By setting this to 0, any data currently traveling 
            // back from DDR (from pre-flush requests) will be ignored 
            // because the response logic checks (pending_req_cnt > 0).
            pending_req_cnt <= 0; 
            
            // 4. Align Address
            // Mask lower 2 bits to get 32-bit aligned word address
            next_fetch_addr <= {start_addr[31:2], 2'b00};
            
            // 5. Set Offset
            // If start_addr is 0x1001, we fetch 0x1000, but start consuming at offset 1.
            // NAC is Big Endian stream mapping to Little Endian bus usually.
            // byte_offset: 0=MSB .. 3=LSB. 
            // We use simple 0..3 mapping here; ordering is handled in Extraction logic.
            byte_offset <= start_addr[1:0];
        end 
        else begin
            // ----------------------------------------------------------------
            // 1. PRODUCER: Request Logic (Prefetcher)
            // ----------------------------------------------------------------
            // Condition to request:
            // 1. We are not currently asserting a request (wait for grant).
            // 2. We have space in FIFO (Current Words + Pending Requests < Depth).
            
            if ( !mem_req && ((words_in_fifo + pending_req_cnt) < FIFO_DEPTH) ) begin
                mem_addr <= next_fetch_addr;
                mem_req  <= 1;
            end

            // Handle Bus Grant (Handshake)
            if (mem_req && mem_grant) begin
                mem_req <= 0; // Deassert strobe, transaction accepted
                pending_req_cnt <= pending_req_cnt + 1; // Mark as in-flight
                next_fetch_addr <= next_fetch_addr + 4; // Advance fetch pointer
            end

            // ----------------------------------------------------------------
            // 2. PRODUCER: Response Logic (Data Ingestion)
            // ----------------------------------------------------------------
            if (mem_valid) begin
                if (pending_req_cnt > 0) begin
                    // Valid data belongs to current stream
                    fifo_data[wr_ptr] <= mem_rdata;
                    
                    // Increment Write Pointer (Circular)
                    wr_ptr <= wr_ptr + 1; 
                    
                    // Update Counters
                    words_in_fifo <= words_in_fifo + 1;
                    pending_req_cnt <= pending_req_cnt - 1;
                end 
                else begin
                    // pending_req_cnt == 0 means this data is "Stale"
                    // (It was requested before the last Flush).
                    // Action: Drop it silently.
                end
            end

            // ----------------------------------------------------------------
            // 3. CONSUMER: Byte Extraction Logic
            // ----------------------------------------------------------------
            byte_valid <= 0;

            if (byte_req && words_in_fifo > 0) begin
                byte_valid <= 1;

                // Extract Byte based on Endianness
                // NAC .b64 stream is treated as Big Endian byte stream.
                // If mem_rdata is [31:0] = 0xAABBCCDD
                // Addr+0 should be AA, Addr+3 should be DD.
                case (byte_offset)
                    2'b00: byte_data <= fifo_data[rd_ptr][31:24]; // Byte 0
                    2'b01: byte_data <= fifo_data[rd_ptr][23:16]; // Byte 1
                    2'b10: byte_data <= fifo_data[rd_ptr][15:8];  // Byte 2
                    2'b11: byte_data <= fifo_data[rd_ptr][7:0];   // Byte 3
                endcase

                // Advance Pointer
                if (byte_offset == 3) begin
                    // Word consumed completely
                    byte_offset <= 0;
                    rd_ptr <= rd_ptr + 1; // Move to next word in FIFO
                    words_in_fifo <= words_in_fifo - 1;
                end else begin
                    // Move to next byte in same word
                    byte_offset <= byte_offset + 1;
                end
            end
        end
    end

endmodule