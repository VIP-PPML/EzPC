#include "Millionaire/millionaire.h"
#include "OT/emp-ot.h"
#include "utils/emp-tool.h"
#include <iostream>

void test_millionaire_protocol(uint64_t alice_value, uint64_t bob_value, int party);

int main(int argc, char **argv) {
    if (argc != 2) {
      printf("Usage: ./millionaire-OT 1 & ./millionaire-OT 2\n");
      return 0;
    }
    int party = atoi(argv[1]);

    // Run the millionaires test with different values for Alice and Bob
    test_millionaire_protocol(2, 3, party);
    test_millionaire_protocol(123456789123456789, 123456789123456788, party);
    test_millionaire_protocol(5, 18000000000000000000, party);
    test_millionaire_protocol(16000000000000000000, 54, party);
    test_millionaire_protocol(4193213593, 6002221524, party);

    return 0;
}

void test_millionaire_protocol(uint64_t alice_value, uint64_t bob_value, int party) {
    int port = 12345;

    sci::IOPack *iopack = new sci::IOPack(party, port);
    sci::OTPack *otpack = new sci::OTPack(iopack, party);

    // Create instance of MillionaireProtocol
    MillionaireProtocol mil_protocol(party, iopack, otpack);

    uint8_t alice_result;
    uint8_t bob_result;

    // Perform the comparison
    if (party == sci::ALICE) {
        mil_protocol.compare(&alice_result, &alice_value, 16, 64);
    } else {
        mil_protocol.compare(&bob_result, &bob_value, 16, 64);
    }

    // Synchronize to ensure both parties have completed the protocol
    iopack->io->flush();

    // Exchange results between parties for XOR operation (secret sharing)
    if (party == sci::ALICE) {
        iopack->io->send_data(&alice_result, sizeof(uint8_t));
        iopack->io->recv_data(&bob_result, sizeof(uint8_t));
    } else {
        iopack->io->recv_data(&alice_result, sizeof(uint8_t));
        iopack->io->send_data(&bob_result, sizeof(uint8_t));
    }

    // Output and check results
    uint8_t xor_result;
    if (party == sci::ALICE) {
      xor_result = alice_result ^ bob_result;
      if (xor_result) {
        std::cout << "Alice > Bob" << std::endl;
      }
      else {
        std::cout << "Bob > Alice" << std::endl;
      }

      // Check if results match expected results
      bool expected_result = alice_value > bob_value;
      if (!(expected_result ^ xor_result)) {
        printf("Test Passed\n");
      }
      else {
        printf("Test Failed\n");
        assert(false);
      }
    }

    // Clean up
    delete iopack;
    delete otpack;

}
