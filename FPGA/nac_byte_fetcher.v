`include "nac_defines.vh"

module NAC_Byte_Fetcher #(
    parameter FIFO_DEPTH_LOG2 = 2, //4 better
    // 0 = Little Endian (Standard AXI/ARM/Zynq) -> Byte 0 is [7:0]
    // 1 = Big Endian (Network order/Legacy)     -> Byte 0 is [31:24]
    parameter BIG_ENDIAN = 0 
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        flush,        
    input  wire [31:0] start_addr,   
    output wire        busy,         
    input  wire        byte_req,     
    output reg  [7:0]  byte_data,    
    output reg         byte_valid,   
    output reg  [31:0] mem_addr,     
    output reg         mem_req,      
    input  wire        mem_grant,    
    input  wire        mem_valid,    
    input  wire [31:0] mem_rdata     
);

    localparam FIFO_DEPTH = (1 << FIFO_DEPTH_LOG2);
    
    reg [31:0] next_fetch_addr;
    reg [31:0] fifo_data [0:FIFO_DEPTH-1];
    reg [FIFO_DEPTH_LOG2-1:0] wr_ptr;
    reg [FIFO_DEPTH_LOG2-1:0] rd_ptr;
    reg [FIFO_DEPTH_LOG2:0]   words_in_fifo; 
    reg [1:0]  byte_offset; 
    reg [FIFO_DEPTH_LOG2:0] pending_req_cnt; 
    
    // Busy if FIFO is empty, indicating pipeline stall
    assign busy = (words_in_fifo == 0);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mem_req <= 0; mem_addr <= 0; byte_valid <= 0; byte_data <= 0;
            wr_ptr <= 0; rd_ptr <= 0; words_in_fifo <= 0; byte_offset <= 0;
            next_fetch_addr <= 0; pending_req_cnt <= 0;
        end 
        else if (flush) begin
            // Flush pipeline on jumps/reset
            mem_req <= 0; byte_valid <= 0;
            wr_ptr <= 0; rd_ptr <= 0; words_in_fifo <= 0;
            pending_req_cnt <= 0; 
            // Align start address to 32-bit word boundary
            next_fetch_addr <= {start_addr[31:2], 2'b00};
            // Capture initial byte offset (0-3)
            byte_offset <= start_addr[1:0];
        end 
        else begin
            // 1. Memory Request Logic (Prefetching)
            // Keep requesting until FIFO is full (considering pending requests)
            if ( !mem_req && ((words_in_fifo + pending_req_cnt) < FIFO_DEPTH) ) begin
                mem_addr <= next_fetch_addr;
                mem_req  <= 1;
            end

            // 2. Handle AXI Handshake (Grant)
            if (mem_req && mem_grant) begin
                mem_req <= 0; 
                pending_req_cnt <= pending_req_cnt + 1; 
                next_fetch_addr <= next_fetch_addr + 4; 
            end

            // 3. Handle Memory Response (Data Valid)
            if (mem_valid) begin
                if (pending_req_cnt > 0) begin
                    fifo_data[wr_ptr] <= mem_rdata;
                    wr_ptr <= wr_ptr + 1; 
                    words_in_fifo <= words_in_fifo + 1;
                    pending_req_cnt <= pending_req_cnt - 1;
                end 
            end

            // 4. Byte Extraction Logic (Consumer)
            byte_valid <= 0;
            if (byte_req && words_in_fifo > 0) begin
                byte_valid <= 1;
                
                // Configurable Endianness Handling
                if (BIG_ENDIAN) begin
                    // Big Endian: Byte 0 is MSB [31:24]
                    case (byte_offset)
                        2'b00: byte_data <= fifo_data[rd_ptr][31:24]; 
                        2'b01: byte_data <= fifo_data[rd_ptr][23:16]; 
                        2'b10: byte_data <= fifo_data[rd_ptr][15:8];  
                        2'b11: byte_data <= fifo_data[rd_ptr][7:0];   
                    endcase
                end else begin
                    // Little Endian (Zynq/AXI Default): Byte 0 is LSB [7:0]
                    case (byte_offset)
                        2'b00: byte_data <= fifo_data[rd_ptr][7:0]; 
                        2'b01: byte_data <= fifo_data[rd_ptr][15:8]; 
                        2'b10: byte_data <= fifo_data[rd_ptr][23:16];  
                        2'b11: byte_data <= fifo_data[rd_ptr][31:24];   
                    endcase
                end

                // Pointer Management
                if (byte_offset == 3) begin
                    byte_offset <= 0;
                    rd_ptr <= rd_ptr + 1; 
                    words_in_fifo <= words_in_fifo - 1;
                end else begin
                    byte_offset <= byte_offset + 1;
                end
            end
        end
    end
endmodule