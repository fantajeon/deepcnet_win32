#ifdef CIFAR100
void readBINFile(vector<Picture*> &characters, const char* filename, bool mirror=false) {
  ifstream file(filename,ios::in|ios::binary);
  if (!file) {
    cout <<"Cannot find " << filename << endl;
    exit(EXIT_FAILURE);
  }
  cout << "\r" << filename;
  unsigned char label[2];
  while (file.read((char*)label,2)) {
    OfflineGridPicture* character = new OfflineGridPicture(32,32,label[1]);
    unsigned char bitmap[3072];
    file.read((char*)bitmap,3072);
    for (int i=0;i<3072;i++)
      character->bitmap[i]=bitmap[i]-128; //Grey == (0,0,0)
    characters.push_back(character);
  }
  file.close();
}
void loadData()
{
  char filenameTrain[]="Data/CIFAR100/train.bin";
  readBINFile(trainCharacters,filenameTrain);
  char filenameTest[]="Data/CIFAR100/test.bin";
  readBINFile(testCharacters,filenameTest);
  cout <<" " << trainCharacters.size()<< " " << testCharacters.size() << endl;
}
#endif
