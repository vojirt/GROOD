annos = load('cars_annos.mat');

classes = [annos.annotations.class];
test = [annos.annotations.test];
cnames = [annos.class_names];

f = fopen('class_names.txt', 'w');
for I=1:length(cnames)
    fprintf(f, '%s\n', cnames{I});
end
fclose(f);

f = fopen('data.txt', 'w');
for I=1:length(test)
    fprintf(f, '%06d.jpg, %d, %d\n', I, classes(I), test(I));
end
fclose(f);
