# $B?J2=E*%b%G%k%^!<%8$N$*;n$7%9%/%j%W%H(B

mergekit$B$r;H$C$??J2=E*%b%G%k%^!<%8$K$h$k(Bllm$B$N@-G=8~>e$r8!>Z$7$^$9!#(B
$B!c;29M(BURL$B!d(Bhttps://note.com/npaka/n/nb9abed7b1ecf

## $BL\<!(B

- [$B4D6-@_Dj(B]$B#1(B(#$B4D6-@_Dj(B)


## $B4D6-@_Dj(B
### ~/.bashrc$B$K0J2<$N4D6-JQ?t$r@_Dj(B

export LANG='ja_JP.UTF-8'
export WANDB_API_KEY="**************************************"
export OPENAI_API_KEY="******************************************"
export HUGGINGFACE_TOKEN="**************************************"
export WANDB_ENTITY="********"
export PATH="$HOME/.local/bin:$PATH"

### $BJQ99$rH?1G(B
source ~/.bashrc

### conda$B$G2>A[4D6-$r%"%/%F%#%Y!<%H(B
#### conda$B$,L$%$%s%9%H!<%k$N>l9g$O!"(Bsetup$B%9%/%j%W%H$N%3%a%s%H%"%&%H$7$F%$%s%9%H!<%k(B
conda activate merge

### $B%;%C%H%"%C%W%9%/%j%W%H$r<B9T(B
bash mergekit-setup.sh
