#!/bin/bash
# 一键更新网页脚本

cd "$(dirname "$0")"

echo "========== 生成HTML =========="
python html_zcq.py --confusion

echo ""
echo "========== 推送到GitHub =========="
git add docs/index.html
git commit -m "Update $(date '+%Y-%m-%d %H:%M')"
git push github-pages main

echo ""
echo "========== 完成 =========="
echo "网页地址: https://zcq991029.github.io/zcq-visualization/"
echo "等待1-2分钟后刷新即可看到更新"
