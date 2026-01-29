import requests
import json
import re
from typing import Dict, Optional, Any, List, BinaryIO
import base64
import os


class FeishuCommentTester:
    """飞书项目评论测试器（支持图片）"""

    def __init__(self):
        # 飞书插件认证信息
        self.PLUGIN_ID = "MII_64EDCCED5EC38003"
        self.PLUGIN_SECRET = "F0B574D7270754A7A4BF4EB60FEBD5C4"
        self.USER_KEY = "7565630504918138881"
        self.endpoint = "https://project.feishu.cn/open_api"
        self.token = None
        self.upload_tokens = {}  # 存储上传令牌 {project_key: token}

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

    def get_upload_token(self, project_key: str) -> str:
        """获取文件上传token"""
        if not self.token:
            self.get_token()

        # 检查是否有缓存的token
        if project_key in self.upload_tokens:
            return self.upload_tokens[project_key]

        print(f"🔑 获取文件上传token...")

        url = f"{self.endpoint}/{project_key}/file/token"
        headers = {
            "Content-Type": "application/json",
            "X-USER-KEY": self.USER_KEY,
            "X-PLUGIN-TOKEN": self.token,
        }

        request_body = {
            "type": 1,  # 1表示评论附件
            "summary": "comment_image"
        }

        try:
            response = requests.post(url, headers=headers, json=request_body)
            response.raise_for_status()

            token_resp = response.json()
            if token_resp.get("err_code", 0) == 0:
                upload_token = token_resp["data"]["token"]
                self.upload_tokens[project_key] = upload_token
                print(f"✅ 成功获取上传token")
                return upload_token
            else:
                print(f"❌ 获取上传token失败: {token_resp.get('err_msg', '未知错误')}")
                raise Exception(f"获取上传token失败: {token_resp.get('err_msg')}")
        except Exception as e:
            print(f"❌ 获取上传tokenAPI失败: {str(e)}")
            raise

    def upload_image(self, project_key: str, image_path: str) -> Optional[Dict[str, Any]]:
        """上传图片到飞书项目

        Returns:
            dict: 包含file_key和file_name的字典，用于评论中的引用
        """
        # 首先检查文件是否存在
        if not os.path.exists(image_path):
            print(f"❌ 图片文件不存在: {image_path}")
            return None

        try:
            # 获取上传token
            upload_token = self.get_upload_token(project_key)

            print(f"📤 上传图片: {image_path}")

            # 准备上传请求
            url = f"{self.endpoint}/file/upload"
            headers = {
                "X-File-Token": upload_token,
            }

            # 读取文件并准备multipart/form-data
            file_name = os.path.basename(image_path)
            with open(image_path, 'rb') as f:
                files = {
                    'file': (file_name, f, 'image/jpeg')  # 可以根据需要调整mime类型
                }
                # 添加其他表单字段
                data = {
                    'file_name': file_name,
                    'parent_type': 1,  # 1表示评论附件
                    'parent_node': ''  # 可以为空
                }

                response = requests.post(url, headers=headers, files=files, data=data)

            print(f"📊 上传响应状态码: {response.status_code}")

            if response.status_code == 200:
                upload_resp = response.json()
                print(f"📋 上传响应: {json.dumps(upload_resp, indent=2, ensure_ascii=False)}")

                if upload_resp.get("err_code", 0) == 0:
                    file_data = upload_resp.get("data", {})
                    file_key = file_data.get("file_key")
                    file_name = file_data.get("file_name")

                    if file_key:
                        print(f"✅ 图片上传成功!")
                        print(f"   - 文件Key: {file_key}")
                        print(f"   - 文件名: {file_name}")
                        print(f"   - 文件类型: {file_data.get('mime_type')}")
                        print(f"   - 文件大小: {file_data.get('size')} bytes")
                        return file_data
                    else:
                        print(f"❌ 上传成功但未返回file_key")
                        return None
                else:
                    print(f"❌ 上传失败: {upload_resp.get('err_msg', '未知错误')}")
                    return None
            else:
                print(f"❌ 上传HTTP错误: {response.status_code}")
                print(f"📄 响应内容: {response.text}")
                return None

        except Exception as e:
            print(f"❌ 上传图片失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def parse_project_url(self, project_url: str) -> Dict[str, str]:
        """解析飞书项目URL，提取各种信息"""
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

    def create_comment_with_image(self, project_key: str, work_item_type_key: str,
                                  work_item_id: str, content: str,
                                  image_file_keys: List[str] = None) -> bool:
        """创建带图片的评论

        Args:
            image_file_keys: 通过upload_image获得的file_key列表
        """
        if not self.token:
            self.get_token()

        print(f"💬 创建带图片的评论...")

        url = f"{self.endpoint}/{project_key}/work_item/{work_item_type_key}/{work_item_id}/comment/create"
        headers = {
            "Content-Type": "application/json",
            "X-USER-KEY": self.USER_KEY,
            "X-PLUGIN-TOKEN": self.token,
        }

        # 构建评论内容
        request_body = {
            "content": content,
        }

        # 如果有图片，添加附件信息
        if image_file_keys:
            request_body["resources"] = [
                {
                    "type": 2,  # 2表示图片
                    "file_key": file_key
                }
                for file_key in image_file_keys
            ]
            print(f"🖼️ 添加 {len(image_file_keys)} 张图片")

        print(f"🔗 请求URL: {url}")
        print(f"📝 评论文本: {content}")
        if image_file_keys:
            print(f"📎 图片keys: {image_file_keys}")
        print(f"📦 请求体: {json.dumps(request_body, ensure_ascii=False)}")

        try:
            response = requests.post(url, headers=headers, json=request_body)

            print(f"📊 响应状态码: {response.status_code}")

            if response.status_code == 200:
                comment_resp = response.json()
                print(f"📋 完整响应: {json.dumps(comment_resp, indent=2, ensure_ascii=False)}")

                if comment_resp.get("err_code", 0) == 0:
                    print(f"✅ 成功创建带图片的评论!")

                    # 修复这里：API直接返回评论ID（int）而不是字典
                    comment_data = comment_resp.get("data")

                    # 判断返回的数据类型
                    if isinstance(comment_data, dict):
                        # 如果是字典，按原方式处理
                        print(f"   - 评论ID: {comment_data.get('comment_id')}")
                        print(f"   - 创建时间: {comment_data.get('created_at')}")

                        resources = comment_data.get('resources', [])
                        if resources:
                            print(f"   - 附件数量: {len(resources)}")
                            for i, resource in enumerate(resources):
                                print(f"     附件{i + 1}: {resource.get('file_name')} ({resource.get('file_key')})")
                    elif isinstance(comment_data, (int, str)):
                        # 如果直接是评论ID（int或str）
                        print(f"   - 评论ID: {comment_data}")
                    else:
                        print(f"   - 返回数据格式: {type(comment_data)}")

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
            import traceback
            traceback.print_exc()
            return False

    def test_comment_with_image(self, project_url: str,
                                comment_text: str = "test with image",
                                image_paths: List[str] = None):
        """测试带图片的评论功能的主方法"""
        try:
            print("🚀 开始测试飞书项目带图片评论功能")
            print("=" * 50)

            # 1. 解析项目URL
            url_info = self.parse_project_url(project_url)
            project_key = url_info['project_key']
            work_item_type_key = url_info['work_item_type_key']
            work_item_id = url_info['work_item_id']

            print()

            # 2. 上传图片（如果有）
            image_file_keys = []
            if image_paths:
                for image_path in image_paths:
                    # 先检查文件是否存在
                    if not os.path.exists(image_path):
                        print(f"❌ 图片文件不存在: {image_path}")
                        print(f"💡 请检查路径是否正确，当前工作目录: {os.getcwd()}")
                        continue

                    file_data = self.upload_image(project_key, image_path)
                    if file_data and file_data.get("file_key"):
                        image_file_keys.append(file_data["file_key"])
                    else:
                        print(f"⚠️  跳过图片 {image_path}，上传失败")

            print()

            # 3. 创建带图片的评论
            success = self.create_comment_with_image(
                project_key, work_item_type_key, work_item_id,
                comment_text, image_file_keys
            )

            print()
            print("=" * 50)
            if success:
                if image_file_keys:
                    print(f"🎉 测试完成! 带{len(image_file_keys)}张图片的评论发送成功!")
                else:
                    print("🎉 测试完成! 纯文本评论发送成功!")
            else:
                print("💥 测试失败! 评论发送失败!")

            return success

        except Exception as e:
            print(f"❌ 测试过程中发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def test_comment(self, project_url: str,
                     comment_text: str = "test"):
        """兼容原方法的测试评论功能"""
        return self.test_comment_with_image(project_url, comment_text, [])


def main():
    """主函数 - 交互式测试（支持图片）"""
    tester = FeishuCommentTester()

    print("🎯 飞书项目评论测试工具（支持图片）")
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

            # 输入评论内容
            comment_content = input("请输入评论内容 (默认: test with image): ").strip()
            if not comment_content:
                comment_content = "test with image"

            # 询问是否添加图片
            add_images = input("是否要添加图片? (y/n, 默认n): ").strip().lower()
            image_paths = []

            if add_images == 'y':
                print("\n💡 提示:")
                print("   - 可以输入绝对路径或相对路径")
                print("   - 可以一次添加多张图片，输入 'done' 完成")
                print("   - 当前工作目录: " + os.getcwd())
                print("-" * 40)

                while True:
                    image_path = input("请输入图片路径 (输入 'done' 完成添加): ").strip()

                    if image_path.lower() == 'done':
                        if image_paths:
                            print(f"✅ 已添加 {len(image_paths)} 张图片")
                        break

                    if not image_path:
                        continue

                    # 检查文件是否存在
                    if os.path.exists(image_path):
                        image_paths.append(image_path)
                        print(f"✅ 添加图片: {image_path}")
                    else:
                        print(f"❌ 文件不存在: {image_path}")
                        print("💡 请检查路径是否正确")

            print()

            # 执行测试
            if image_paths:
                tester.test_comment_with_image(project_url, comment_content, image_paths)
            else:
                tester.test_comment(project_url, comment_content)

        except KeyboardInterrupt:
            print("\n\n👋 用户中断，再见!")
            break
        except Exception as e:
            print(f"❌ 发生错误: {str(e)}")


if __name__ == "__main__":
    # 可以直接运行测试，也可以作为模块导入

    # # 示例1：直接测试特定URL（纯文本）
    # tester = FeishuCommentTester()
    # test_url = "https://project.feishu.cn/iffcom/case/detail/6644337846"
    # tester.test_comment(test_url, "test with image")

    # 示例2：测试带图片的评论
    tester = FeishuCommentTester()
    image_paths = [os.path.join("/get_data/draw_image/87220060/1767085693900000/avm.jpg")]
    test_url = "https://project.feishu.cn/iffcom/case/detail/6644337846"
    tester.test_comment_with_image(test_url, "请看这些图片", image_paths)