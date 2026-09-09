#include <stdio.h>
#include <string.h>

int main() {
    char data[13], stuffed[30], destuffed[30];
    char flag[] = "01111110";
    int i, j = 0, k = 0, count = 0;

    printf("Enter bit stream (maximum 12 bits): ");
    scanf("%12s", data);

    // Check that input contains only 0 and 1
    for (i = 0; data[i] != '\0'; i++) {
        if (data[i] != '0' && data[i] != '1') {
            printf("Invalid input! Enter only 0 and 1.\n");
            return 0;
        }
    }

    // Bit Stuffing
    for (i = 0; data[i] != '\0'; i++) {
        stuffed[j++] = data[i];

        if (data[i] == '1')
            count++;
        else
            count = 0;

        if (count == 5) {
            stuffed[j++] = '0';
            count = 0;
        }
    }

    stuffed[j] = '\0';

    // De-stuffing
    count = 0;

    for (i = 0; stuffed[i] != '\0'; i++) {
        destuffed[k++] = stuffed[i];

        if (stuffed[i] == '1')
            count++;
        else
            count = 0;

        if (count == 5) {
            i++;       // Remove stuffed 0
            count = 0;
        }
    }

    destuffed[k] = '\0';

    // Display results
    printf("\n--- Bit Stuffing and De-stuffing ---\n");

    printf("Flag            : %s\n", flag);
    printf("Original Data   : %s\n", data);

    printf("After Stuffing  : %s%s%s\n",
           flag, stuffed, flag);

    printf("After De-stuffing: %s%s%s\n",
           flag, destuffed, flag);

    return 0;
}