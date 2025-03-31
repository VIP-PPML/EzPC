#include "Millionaire/millionaire.h"
#include "OT/emp-ot.h"
#include "utils/emp-tool.h"
#include <iostream>

int main(int argc, char **argv) {
    int port = 12345;
    int party = atoi(argv[1]);

    sci::IOPack *iopack = new sci::IOPack(party, port);
    sci::OTPack *otpack = new sci::OTPack(iopack, party);

    // Create instances of MillionaireProtocol for both parties
    MillionaireProtocol alice(sci::ALICE, iopack, otpack);
    MillionaireProtocol bob(sci::BOB, iopack, otpack);

    // Define the values to compare
    uint64_t alice_value = 123456789123456789;
    uint64_t bob_value = 123456789123456788;

    uint8_t alice_result;
    uint8_t bob_result;

    // Perform the comparison
    if (party == sci::ALICE) {
        alice.compare(&alice_result, &alice_value, 16, 64);
    } else {
        bob.compare(&bob_result, &bob_value, 16, 64);
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

    // Output results 
    uint8_t xor_result;
    if (party == sci::ALICE) {
      xor_result = alice_result ^ bob_result;
      if (xor_result) {
        std::cout << "Alice > Bob" << std::endl;
      }
      else {
        std::cout << "Bob > Alice" << std::endl;
      }
    }

    // Clean up
    delete iopack;
    delete otpack;

    return 0;
}
