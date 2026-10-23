"""
This module contains a function to print seperate lane.
Yuntao Li, 2/8/2024
"""

def print_util(*headings):
    for i, heading in enumerate(headings, start=1):
        if heading == 'h1':
            print('\n' + '=' * 80 + '\n')
        elif heading == 'h2':
            print('\n' + '=' * 60 + '\n')
        elif heading == 'h3':
            print('\n' + '=' * 40 + '\n')
        elif heading == 'h4':
            print('\n' + '=' * 20 + '\n')
        else:
            print('Invalid heading:', heading)

# Example usage
# print_util('h1')

