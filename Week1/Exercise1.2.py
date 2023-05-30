import prettytable, re, collections

table = prettytable.PrettyTable(field_names = ["NAME", "FREQUENCY"])

[table.add_row([char, count]) for char, count in collections.Counter(re.findall(r'\w', input("Please enter your text: "))).items()]
    
print(table)