# Definition Function: 
def table_counter_char(text):
    
    counter = {}
    for char in text:
        if char in counter and char.isalnum():
            counter[char] += 1
        elif not char.isalnum():
            continue
        else:
            counter[char] = 1

    # Plot Table:
    print('+------+-----------+')
    print('| NAME | FREQUENCY |')
    print('+======+===========+')
    for char, count in counter.items():
        print(f'|   {char}  |     {count}     |')
        print('+------+-----------+')

# Test Output
context = input('Please Enter your context: ')
table_counter_char(context)