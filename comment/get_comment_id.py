#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
飞书项目评论测试工具

功能：
1. 解析飞书项目URL
2. 搜索指定用户获取用户信息
3. 发送带有@提及功能的评论

作者：自动化脚本
版本：1.0
"""

import requests
import json
import re
from typing import Optional, Dict, Any
from urllib.parse import urlparse


class FeishuCommentTester:
    """飞书项目评论测试器"""
    
    def __init__(self):
        # 飞书插件认证信息
        self.PLUGIN_ID = "MII_64EDCCED5EC38003"
        self.PLUGIN_SECRET = "F0B574D7270754A7A4BF4EB60FEBD5C4"
        self.USER_KEY = "7565630504918138881"
        self.endpoint = "https://project.feishu.cn/open_api"
        self.token = None
    
    def get_token(self):
        """获取飞书API访问token"""
        url = f"{self.endpoint}/authen/plugin_token"
        request_body = {
            "plugin_id": self.PLUGIN_ID,
            "plugin_secret": self.PLUGIN_SECRET,
            "type": 0,
        }
        
        response = requests.post(url, json=request_body)
        response.raise_for_status()
        token_resp = response.json()
        self.token = token_resp["data"]["token"]
        print(f"✅ 成功获取token")
        return self.token
    
    def parse_project_url(self, project_url: str) -> Dict[str, str]:
        """解析飞书项目URL，提取各种信息
        
        支持多种URL格式：
        - https://project.feishu.cn/{project_key}/case/detail/{work_item_id}
        - https://project.feishu.cn/{project_key}/work_item/{work_item_type_key}/{work_item_id}
        - https://project.feishu.cn/{project_key}/story/detail/{work_item_id}
        - 等等
        """
        print(f"🔍 解析项目URL: {project_url}")
        
        # 通用URL解析模式
        patterns = [
            # case详情页面
            (r'https://project\.feishu\.cn/([^/]+)/case/detail/(\d+)', 'case', '681329d725ac1e8647ae80bd'),
            # story详情页面  
            (r'https://project\.feishu\.cn/([^/]+)/story/detail/(\d+)', 'story', 'story_type_key'),
            # 通用work_item URL
            (r'https://project\.feishu\.cn/([^/]+)/work_item/([^/]+)/(\d+)', 'work_item', None),
            # 缺陷详情页面
            (r'https://project\.feishu\.cn/([^/]+)/bug/detail/(\d+)', 'bug', 'bug_type_key'),
        ]
        
        for pattern, item_type, default_type_key in patterns:
            match = re.search(pattern, project_url)
            if match:
                if item_type == 'work_item':
                    project_key = match.group(1)
                    work_item_type_key = match.group(2)
                    work_item_id = match.group(3)
                else:
                    project_key = match.group(1)
                    work_item_id = match.group(2)
                    work_item_type_key = default_type_key
                
                result = {
                    'project_key': project_key,
                    'work_item_type_key': work_item_type_key,
                    'work_item_id': work_item_id,
                    'item_type': item_type
                }
                
                print(f"✅ 成功解析URL:")
                print(f"   - Project Key: {project_key}")
                print(f"   - Work Item Type Key: {work_item_type_key}")
                print(f"   - Work Item ID: {work_item_id}")
                print(f"   - Item Type: {item_type}")
                
                return result
        
        raise ValueError(f"❌ 无法解析URL格式: {project_url}")
    
    def search_user(self, user_name: str, project_key: str) -> Optional[Dict[str, Any]]:
        """搜索用户信息 - 使用/user/search API"""
        if not self.token:
            self.get_token()
        
        print(f"🔍 搜索用户: {user_name}")
        
        url = f"{self.endpoint}/user/search"
        headers = {
            "Content-Type": "application/json",
            "X-USER-KEY": self.USER_KEY,
            "X-PLUGIN-TOKEN": self.token,
        }
        
        request_body = {
            "query": user_name,
        }
        
        try:
            response = requests.post(url, headers=headers, json=request_body)
            response.raise_for_status()
            
            search_resp = response.json()
            if search_resp.get("err_code", 0) == 0:
                users = search_resp.get("data", [])
                if users:
                    user = users[0]
                    print(f"✅ 找到用户:")
                    # name字段是个对象，取default值
                    user_name_display = user.get('name', {}).get('default', 'N/A')
                    print(f"   - 用户名: {user_name_display}")
                    print(f"   - 用户ID: {user.get('user_key', 'N/A')}")
                    print(f"   - 邮箱: {user.get('email', 'N/A')}")
                    print(f"   - 用户名(中文): {user.get('name_cn', 'N/A')}")
                    return user
                else:
                    print(f"❌ 未找到用户: {user_name}")
                    return None
            else:
                print(f"❌ 搜索失败: {search_resp.get('err_msg', '未知错误')}")
                return None
        except Exception as e:
            print(f"❌ 用户搜索API失败: {str(e)}")
            return None
    
    def create_comment_with_mention(self, project_key: str, work_item_type_key: str, 
                                  work_item_id: str,
                                  mention_user_key: str, mention_user_name: str) -> bool:
        """创建评论 (暂时不使用@提及功能)"""
        if not self.token:
            self.get_token()
        
        print(f"💬 创建评论...")
        
        url = f"{self.endpoint}/{project_key}/work_item/{work_item_type_key}/{work_item_id}/comments"
        headers = {
            "Content-Type": "application/json",
            "X-USER-KEY": self.USER_KEY,
            "X-PLUGIN-TOKEN": self.token,
        }
        
        # 使用正确的@用户格式: <at user_id="用户ID">用户名</at>
        # comment_text = f'<at user_id="{mention_user_key}">{mention_user_name}</at> {content}'
        # 暂时不使用@人功能，只发送纯文本
        
        # content字段直接是字符串文本
        request_body = None
        
        print(f"🔗 请求URL: {url}")
        print(f"📦 请求体: {json.dumps(request_body, ensure_ascii=False)}")
        
        try:
            response = requests.get(url, headers=headers)
            
            print(f"📊 响应状态码: {response.status_code}")
            
            if response.status_code == 200:
                comment_resp = response.json()
                print(f"📋 完整响应: {json.dumps(comment_resp, indent=2, ensure_ascii=False)}")
                
                if comment_resp.get("err_code", 0) == 0:
                    print(f"✅ 成功创建评论!")
                    return True
                else:
                    print(f"❌ 创建评论失败: {comment_resp.get('err_msg', '未知错误')}")
                    print(f"错误代码: {comment_resp.get('err_code')}")
                    return False
            else:
                print(f"❌ HTTP错误: {response.status_code}")
                print(f"📄 响应内容: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ 请求异常: {str(e)}")
            return False
    
    def test_comment(self, project_url: str,):
        """测试评论功能的主方法"""
        try:
            print("🚀 开始测试飞书项目评论功能")
            print("=" * 50)
            
            # 1. 解析项目URL
            url_info = self.parse_project_url(project_url)
            project_key = url_info['project_key']
            work_item_type_key = url_info['work_item_type_key']
            work_item_id = url_info['work_item_id']
            
            print()
            
            # 2. 搜索用户 (暂时注释掉，因为不使用@人功能)
            # user_info = self.search_user(mention_user_name, project_key)
            # if not user_info:
            #     print(f"❌ 无法找到用户 {mention_user_name}，停止执行")
            #     return False
            
            # user_key = user_info.get('user_key')
            # # name字段是个对象，优先使用default值，fallback到name_cn或原始输入
            # user_name = (user_info.get('name', {}).get('default') or 
            #             user_info.get('name_cn') or 
            #             mention_user_name)
            
            print()
            
            # 3. 创建评论 (暂时不使用@人功能)
            success = self.create_comment_with_mention(
                project_key, work_item_type_key, work_item_id,
                "", ""  # 传入空字符串，因为暂时不@人
            )
            
            print()
            print("=" * 50)
            if success:
                print("🎉 测试完成! 评论发送成功!")
            else:
                print("💥 测试失败! 评论发送失败!")
                
            return success
            
        except Exception as e:
            print(f"❌ 测试过程中发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """主函数 - 交互式测试"""
    tester = FeishuCommentTester()
    
    print("🎯 飞书项目评论测试工具")
    print("=" * 50)
    
    while True:
        try:
            # 获取用户输入
            project_url = input("\n请输入飞书项目URL (或输入 'quit' 退出): ").strip()
            
            if project_url.lower() in ['quit', 'exit', 'q']:
                print("👋 再见!")
                break
            
            if not project_url:
                print("❌ URL不能为空，请重新输入")
                continue
            
            # 可选：自定义提及用户 (暂时注释掉，因为不使用@人功能)
            # mention_user = input("请输入要@的用户名 (默认: Ruihao Zhao): ").strip()
            # if not mention_user:
            #     mention_user = "Ruihao Zhao"
            mention_user = ""  # 暂时设为空
            
            # 可选：自定义评论内容
            comment_content = input("请输入评论内容 (默认: test): ").strip()
            if not comment_content:
                comment_content = "test"
            
            # 执行测试
            tester.test_comment(project_url, mention_user, comment_content)
            
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，再见!")
            break
        except Exception as e:
            print(f"❌ 发生错误: {str(e)}")


if __name__ == "__main__":
    # 可以直接运行测试，也可以作为模块导入
    
    # 示例：直接测试特定URL
    tester = FeishuCommentTester()
    test_url = "https://project.feishu.cn/iffcom/case/detail/6730438119"
    tester.test_comment(test_url,)
    
    # 交互式模式
    # main()
